#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: lluvia estimada por satélite (NOAA Enterprise Rain Rate, GOES-19)  ·  v2
#
# Qué hace, en orden:
#   1. Busca en el bucket público de NOAA el archivo más reciente (uno cada 10 min, ~68 MB).
#   2. Descarta el escaneo si es demasiado viejo (EDAD_MAX_MIN) en vez de guardarlo sin avisar.
#   3. Lo abre SIN descargarlo entero (solo lee los trozos que necesita) y cierra todo al terminar.
#   4. Muestrea la lluvia (mm/h) en una rejilla de puntos, distinguiendo "0.0 = no llueve" de
#      "null = sin información" (fuera de vista, calidad mala, valor inválido).
#   5. Guarda la rejilla en Neon (tabla lluvia_sat) para que la app la lea vía /api/lluvia-sat.
#
# Variable obligatoria para guardar: DATABASE_URL (si falta, solo muestra el resultado = modo prueba).
# Variables opcionales (todas con valor por defecto; vacías = se usa el defecto):
#   LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, PASO   -> área y separación de la rejilla (grados)
#   EDAD_MAX_MIN                               -> antigüedad máxima del escaneo (minutos)
#   BUCKET, PREFIJO                            -> origen de los datos
# Uso de prueba:  python scripts/sat_lluvia.py
# =====================================================
import json
import logging
import os
import re
import sys
import tempfile
import datetime as dt
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from urllib.parse import urlparse

import numpy as np
import requests
import xarray as xr

log = logging.getLogger('sat_lluvia')


# ---------- Configuración (todo se puede cambiar sin editar el código) ----------
def _env(nombre, defecto, tipo=str):
    """Lee una variable de entorno; si no existe o está vacía, devuelve el valor por defecto."""
    valor = os.environ.get(nombre)
    return tipo(valor) if valor not in (None, '') else defecto


BUCKET = _env('BUCKET', 'https://noaa-enterprise-rainrate-pds.s3.amazonaws.com')
PREFIJO = _env('PREFIJO', 'G19/ABI-L2-FLDK-RainfallRate')     # GOES-19, disco completo
# Área muestreada: Guayaquil, Samborondón, Durán, Pascuales y Vía a la Costa (hasta ~Chongón)
LAT_MIN, LAT_MAX = _env('LAT_MIN', -2.36, float), _env('LAT_MAX', -2.00, float)
LON_MIN, LON_MAX = _env('LON_MIN', -80.14, float), _env('LON_MAX', -79.76, float)
PASO = _env('PASO', 0.02, float)                               # separación entre puntos (~2.2 km)
EDAD_MAX_MIN = _env('EDAD_MAX_MIN', 40.0, float)               # más viejo que esto = no se guarda
HORAS_BUSQUEDA = 4                                             # cuántas horas hacia atrás se busca un archivo
# REGLAS FIJAS (no son configurables a propósito: cambiarlas cambia el significado de los datos de alerta):
#  - El cielo despejado llega como 0.0 con DQF 0 (verificado en el diagnóstico del disco completo). El relleno es -999.0:
#    un valor negativo significa "sin dato", nunca "no llueve".
#  - DQF NO es una lista de códigos: es una máscara de bits. El bit 0 = "recuperación de lluvia mala" (valor 1).
#    Verificado: los códigos con ese bit (3) traen -999; los demás (0, 2, 64, 66) traen lluvia real, así que se aceptan.
BIT_RECUPERACION_MALA = 1
MM_MAX_VALIDO = 300.0                                          # margen amplio: un valor mayor es un error, no una tormenta
MAX_PUNTOS = 5000                                              # tope de seguridad del tamaño de la rejilla
# Rutas posibles del bucket: calendario (AAAA/MM/DD/HH) o día juliano (AAAA/JJJ/HH)
FORMATOS_RUTA = ('%Y/%m/%d/%H', '%Y/%j/%H')
# Valores por defecto de la órbita de GOES-Este (solo si el archivo no trae su proyección)
PROY_DEFECTO = {'req': 6378137.0, 'rpol': 6356752.31414, 'h': 35786023.0, 'lam0': -75.2}


# ---------- 1. Buscar el archivo más reciente ----------
def listar_ultimo():
    """Devuelve la clave del archivo más nuevo; mira la hora UTC actual y hasta HORAS_BUSQUEDA-1 horas atrás.
    Prueba las dos formas de carpeta y deja en el log cuál funcionó."""
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    ahora = dt.datetime.now(dt.timezone.utc)
    for atras in range(HORAS_BUSQUEDA):
        h = ahora - dt.timedelta(hours=atras)
        for formato in FORMATOS_RUTA:
            prefijo = f'{PREFIJO}/{h.strftime(formato)}/'
            r = requests.get(BUCKET + '/', params={'list-type': '2', 'prefix': prefijo}, timeout=30)
            r.raise_for_status()
            claves = [e.text for e in ET.fromstring(r.content).findall('s3:Contents/s3:Key', ns)
                      if e.text and e.text.endswith('.nc')]
            if claves:
                log.info('Carpeta con formato %s', formato)
                return sorted(claves)[-1]                      # el nombre incluye la hora: el último es el más nuevo
    raise RuntimeError(f'No hay archivos en las últimas {HORAS_BUSQUEDA} horas')


def hora_escaneo(clave):
    """Lee la hora de inicio del escaneo desde el nombre. Acepta _sAAAAMMDDhhmmssd_ (15 dígitos)
    y _sAAAAJJJhhmmssd_ (14 dígitos, JJJ = día del año)."""
    m = re.search(r'_s(\d{14,15})_', clave)
    if not m:
        raise RuntimeError('No se pudo leer la hora en el nombre: ' + clave)
    d, utc = m.group(1), dt.timezone.utc
    if len(d) == 15:
        return dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12]), int(d[12:14]), tzinfo=utc)
    inicio = dt.datetime(int(d[0:4]), 1, 1, int(d[7:9]), int(d[9:11]), int(d[11:13]), tzinfo=utc)
    return inicio + dt.timedelta(days=int(d[4:7]) - 1)


# ---------- 2. Abrir el archivo (y cerrarlo siempre) ----------
@contextmanager
def abrir(clave):
    """Abre el NetCDF leyendo por partes; si no se puede, lo descarga a un archivo temporal ÚNICO.
    Al salir del bloque 'with' cierra el dataset y el archivo, y borra el temporal."""
    url = f'{BUCKET}/{clave}'
    f = ds = tmp = None
    try:
        try:
            import fsspec                                          # necesita aiohttp para leer por HTTP
            f = fsspec.open(url, 'rb', block_size=4 * 1024 * 1024).open()
            ds = xr.open_dataset(f, engine='h5netcdf')
            log.info('Lectura: POR PARTES (fsspec)')
        except (ImportError, OSError, ValueError) as e:
            log.warning('Lectura por partes falló (%s: %s); se descarga el archivo completo', type(e).__name__, str(e)[:120])
            if f is not None:
                f.close()
                f = None
            tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)   # nombre único: no choca con otro proceso
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                for trozo in r.iter_content(1 << 20):
                    tmp.write(trozo)
            tmp.close()
            ds = xr.open_dataset(tmp.name, engine='h5netcdf')
            log.info('Lectura: DESCARGA COMPLETA')
        yield ds
    finally:
        if ds is not None:
            ds.close()
        if f is not None:
            f.close()
        if tmp is not None:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ---------- 3. Muestrear la lluvia ----------
def variable_lluvia(ds):
    """Encuentra la variable con la tasa de lluvia (normalmente 'RRQPE')."""
    for n in ('RRQPE', 'RainfallRate', 'rain_rate', 'RR'):
        if n in ds.data_vars:
            return n
    for n, v in ds.data_vars.items():                       # respaldo: primera variable con ejes x e y
        if 'x' in v.dims and 'y' in v.dims and not n.upper().startswith('DQF'):
            return n
    raise RuntimeError('No encontré la variable de lluvia. Variables: ' + ', '.join(ds.data_vars))


def variable_calidad(ds):
    """Encuentra la variable de calidad (DQF) si el archivo la trae; si no, devuelve None."""
    for n, v in ds.data_vars.items():
        if n.upper().startswith('DQF') and 'x' in v.dims and 'y' in v.dims:
            return n
    return None


def parametros(ds):
    """Datos de la proyección del satélite (vienen dentro del archivo)."""
    a = ds['goes_imager_projection'].attrs if 'goes_imager_projection' in ds else {}
    req = float(a.get('semi_major_axis', PROY_DEFECTO['req']))
    return {
        'req': req,
        'rpol': float(a.get('semi_minor_axis', PROY_DEFECTO['rpol'])),
        'h': float(a.get('perspective_point_height', PROY_DEFECTO['h'])) + req,   # distancia al centro de la Tierra
        'lam0': float(a.get('longitude_of_projection_origin', PROY_DEFECTO['lam0'])),
    }


def proyectar(lat, lon, p):
    """Latitud/longitud -> ángulo de escaneo (x, y en radianes), VECTORIZADO con NumPy (fórmulas del manual GOES-R).
    Devuelve también 'visible': False si el punto no se ve desde el satélite (comprobación del propio manual,
    que reemplaza a un rango de longitudes fijo porque vale para cualquier área)."""
    phi, lam, lam0 = np.radians(lat), np.radians(lon), np.radians(p['lam0'])
    req, rpol, H = p['req'], p['rpol'], p['h']
    e2 = (req ** 2 - rpol ** 2) / req ** 2
    phic = np.arctan((rpol ** 2 / req ** 2) * np.tan(phi))
    rc = rpol / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    sx = H - rc * np.cos(phic) * np.cos(lam - lam0)
    sy = -rc * np.cos(phic) * np.sin(lam - lam0)
    sz = rc * np.sin(phic)
    visible = H * (H - sx) >= sy ** 2 + (req ** 2 / rpol ** 2) * sz ** 2
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return x, y, visible


def indice_en_eje(eje, valores):
    """Posición (entero) de cada valor en un eje regular, sin bucles. -1 si no se puede calcular."""
    paso = (eje[-1] - eje[0]) / (len(eje) - 1)
    if np.abs(np.diff(eje) - paso).max() > abs(paso) * 0.01:
        raise ValueError('El eje del satélite no es regular; no se puede calcular el índice directo')
    idx = np.rint((valores - eje[0]) / paso)
    return np.where(np.isfinite(idx), idx, -1).astype(int)


def muestrear(ds):
    """Devuelve (rejilla, estadísticas). Fila 0 = latitud más al sur, columna 0 = longitud más al oeste.
    En 'mm': número = lluvia (0.0 = hay dato y no llueve); None = SIN INFORMACIÓN."""
    nombre, nom_dqf, p = variable_lluvia(ds), variable_calidad(ds), parametros(ds)
    xs, ys = ds['x'].values.astype('float64'), ds['y'].values.astype('float64')
    lats = np.arange(LAT_MIN, LAT_MAX + PASO / 2, PASO)
    lons = np.arange(LON_MIN, LON_MAX + PASO / 2, PASO)
    if lats.size * lons.size > MAX_PUNTOS:
        raise ValueError(f'La rejilla tendría {lats.size * lons.size} puntos (máximo {MAX_PUNTOS}). Revisa el área o PASO.')
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')

    # Píxel del archivo más cercano a cada punto (todo de una vez) y comprobación de que cae dentro de la imagen
    with np.errstate(invalid='ignore', divide='ignore'):
        x, y, visible = proyectar(LAT, LON, p)
        ix, iy = indice_en_eje(xs, x), indice_en_eje(ys, y)
    dentro = visible & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    if not dentro.any():
        raise RuntimeError('Ningún punto de la rejilla cae dentro de la imagen del satélite. Revisa el área.')

    # Se lee solo el rectángulo que contiene esos píxeles (pocos KB), no toda la imagen
    y0, y1, x0, x1 = iy[dentro].min(), iy[dentro].max(), ix[dentro].min(), ix[dentro].max()
    jy, jx = np.clip(iy, y0, y1) - y0, np.clip(ix, x0, x1) - x0     # los puntos fuera se anclan al borde y se anulan después

    def leer(var):
        bloque = ds[var].isel(y=slice(int(y0), int(y1) + 1), x=slice(int(x0), int(x1) + 1)).transpose('y', 'x').values
        return bloque.astype('float64')[jy, jx]

    valores = leer(nombre)
    calidad = leer(nom_dqf) if nom_dqf else None

    # Clasificación de cada punto: ¿hay dato fiable o no?
    nan = ~np.isfinite(valores)
    absurdo = ~nan & ((valores < 0) | (valores >= MM_MAX_VALIDO))
    if calidad is not None:
        cod = np.where(np.isfinite(calidad), calidad, 0).astype(int)
        dqf_malo = ~np.isfinite(calidad) | ((cod & BIT_RECUPERACION_MALA) != 0)    # sin código, o bit 0 encendido
    else:
        dqf_malo = np.zeros(valores.shape, dtype=bool)
    sin_dato = ~dentro | dqf_malo | absurdo | nan                   # un NaN nunca se convierte en 0.0
    mm = np.where(sin_dato, np.nan, valores)                        # NaN aquí solo marca "sin dato" (luego será null en el JSON)

    stats = {
        'puntos': int(mm.size), 'con_dato': int((~sin_dato).sum()), 'fuera_de_vista': int((~dentro).sum()),
        'dqf_malo': int((dentro & dqf_malo).sum()), 'absurdo': int((dentro & absurdo).sum()), 'nan': int((dentro & nan).sum()),
        'dqf_hist': {str(k): int(v) for k, v in zip(*np.unique(calidad[dentro], return_counts=True))} if calidad is not None else None,
        'dqf_var': nom_dqf,
    }
    rejilla = {
        'variable': nombre, 'dqf': nom_dqf,
        'lat0': round(float(lats[0]), 4), 'lon0': round(float(lons[0]), 4), 'paso': PASO,
        'filas': int(lats.size), 'cols': int(lons.size),
        'cobertura': round(stats['con_dato'] / stats['puntos'], 3),    # fracción de puntos con información
        'mm': [[None if np.isnan(v) else round(float(v), 1) for v in fila] for fila in mm],
    }
    return rejilla, stats


# ---------- 4. Guardar en Neon ----------
def guardar(escaneo, rejilla):
    """Guarda (reemplaza) la última rejilla usando la API SQL por HTTP de Neon, igual que mareas.js."""
    url = os.environ.get('DATABASE_URL')
    if not url:
        log.info('Sin DATABASE_URL: no se guarda (modo prueba).')
        return
    host = urlparse(url).hostname
    r = requests.post(
        f'https://{host}/sql',
        headers={'Content-Type': 'application/json', 'Neon-Connection-String': url},
        json={
            'query': 'INSERT INTO lluvia_sat (id, escaneo, rejilla) VALUES (1, $1::timestamptz, $2::jsonb) '
                     'ON CONFLICT (id) DO UPDATE SET escaneo = EXCLUDED.escaneo, rejilla = EXCLUDED.rejilla, actualizado_en = now()',
            'params': [escaneo.isoformat(), json.dumps(rejilla, allow_nan=False)],   # allow_nan=False: nunca sale un NaN inválido
        },
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f'Neon respondió {r.status_code}: {r.text[:300]}')
    log.info('Guardado en Neon.')


def main():
    clave = listar_ultimo()
    escaneo = hora_escaneo(clave)
    edad = (dt.datetime.now(dt.timezone.utc) - escaneo).total_seconds() / 60
    log.info('Archivo: %s | Escaneo: %s (hace %.0f min)', clave, escaneo.isoformat(), edad)

    # El dato viejo NO se guarda: en Neon queda el último bueno y la app ya lo descarta por su antigüedad
    if edad > EDAD_MAX_MIN:
        log.warning('Escaneo desactualizado (%.0f min > %.0f). No se guarda.', edad, EDAD_MAX_MIN)
        print(f'::warning title=Satélite desactualizado::El archivo más reciente tiene {edad:.0f} min (límite {EDAD_MAX_MIN:.0f}).')
        return

    with abrir(clave) as ds:
        log.info('Variables: %s | dimensiones: %s', list(ds.data_vars), dict(ds.sizes))   # sirve para depurar
        rejilla, st = muestrear(ds)
    rejilla['generado'] = dt.datetime.now(dt.timezone.utc).isoformat()   # el frontend calcula la edad con 'escaneo'

    con_dato = [v for fila in rejilla['mm'] for v in fila if v is not None]
    log.info('Variable de lluvia: %s | calidad: %s', rejilla['variable'], st['dqf_var'] or 'NO EXISTE')
    log.info('Puntos: %d | con dato: %d (%.0f%%) | sin dato: fuera de vista=%d, calidad mala=%d, valor absurdo=%d | NaN=%d',
             st['puntos'], st['con_dato'], rejilla['cobertura'] * 100, st['fuera_de_vista'], st['dqf_malo'], st['absurdo'], st['nan'])
    log.info('Códigos DQF en el área: %s | se descarta si el bit 0 está encendido', st['dqf_hist'])
    log.info('Puntos con lluvia >= 2 mm/h: %d | máximo: %.1f mm/h', sum(v >= 2 for v in con_dato), max(con_dato, default=0.0))
    guardar(escaneo, rejilla)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('Falló la ejecución')       # imprime la traza completa con la línea exacta
        sys.exit(1)
