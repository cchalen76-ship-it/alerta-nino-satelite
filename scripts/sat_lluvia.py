#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: lluvia estimada por satélite (NOAA Enterprise Rain Rate, GOES-19)  ·  v3
#
# Qué hace, en orden:
#   1. Busca en el bucket público de NOAA el archivo más reciente (uno cada 10 min, ~68 MB).
#   2. Descarta el escaneo si es demasiado viejo (EDAD_MAX_MIN) en vez de guardarlo sin avisar.
#   3. Lo abre SIN descargarlo entero (solo lee los trozos que necesita) y cierra todo al terminar.
#   4. Muestrea la lluvia (mm/h) en una rejilla de puntos, distinguiendo "0.0 = no llueve" de
#      "null = sin información" (fuera de vista, calidad mala, valor inválido).
#   5. Guarda la rejilla en Neon (tabla lluvia_sat) para que la app la lea vía /api/lluvia-sat.
#
# Novedades de la v3 (NO cambian lo que se guarda en Neon):
#   - Diagnóstico de calidad en el log: significado de los códigos DQF/PQI según el propio archivo y,
#     por cada código, cuántos puntos hay, cuántos con lluvia y el máximo. Sirve para decidir con evidencia
#     si algún código de calidad debe descartarse.
#   - Estadísticas calculadas con NumPy (sin recorrer listas) y reintentos ante fallos de red (listado y Neon).
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
import time
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
#    OJO: "traen lluvia real" significa que traen un valor numérico, no que sea fiable. El diagnóstico de la v3 (log)
#    muestra qué significa cada código y cuánta lluvia reporta cada uno.
BIT_RECUPERACION_MALA = 1
MM_MAX_VALIDO = 300.0                                          # margen amplio: un valor mayor es un error, no una tormenta
MAX_PUNTOS = 5000                                              # tope de seguridad del tamaño de la rejilla
UMBRAL_LOG_MM = 2.0                                            # solo para estadísticas del log: mismo umbral de llovizna que usa la app
TOP_CODIGOS_LOG = 12                                           # máximo de códigos de calidad que se listan en el log
# Rutas posibles del bucket: calendario (AAAA/MM/DD/HH) o día juliano (AAAA/JJJ/HH)
FORMATOS_RUTA = ('%Y/%m/%d/%H', '%Y/%j/%H')
# Este producto NO trae su proyección, así que se usan estos valores. VERIFICADO contra las Latitude/Longitude del archivo:
# sus coordenadas corresponden a un satélite en -75.0° (con -75.2° el error era de 0.2° de longitud, ~11 píxeles).
PROY_DEFECTO = {'req': 6378137.0, 'rpol': 6356752.31414, 'h': 35786023.0, 'lam0': -75.0}
OFF_REJILLA = 0.151844                                         # ángulo (rad) del borde de la rejilla fija GOES-R de disco completo
TOLERANCIA_MAPEO_GRADOS = 0.06                                 # si el píxel elegido se aleja más que esto del punto pedido, se aborta


# ---------- 0. Utilidad de red ----------
def con_reintento(fn, intentos=3, espera=2.0):
    """Ejecuta fn(); si falla por un error de red (timeout, conexión, HTTP), reintenta con espera creciente.
    El último fallo se propaga tal cual. Los errores de lógica (no son de red) no se reintentan."""
    for n in range(1, intentos + 1):
        try:
            return fn()
        except requests.RequestException as e:
            if n == intentos:
                raise
            log.warning('Fallo de red (%s); reintento %d/%d en %.0f s', type(e).__name__, n, intentos - 1, espera * n)
            time.sleep(espera * n)


# ---------- 1. Buscar el archivo más reciente ----------
def listar_ultimo():
    """Devuelve la clave del archivo más nuevo; mira la hora UTC actual y hasta HORAS_BUSQUEDA-1 horas atrás.
    Prueba las dos formas de carpeta y deja en el log cuál funcionó."""
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    ahora = dt.datetime.now(dt.timezone.utc)

    def pedir(prefijo):
        r = requests.get(BUCKET + '/', params={'list-type': '2', 'prefix': prefijo}, timeout=30)
        r.raise_for_status()
        return r.content

    for atras in range(HORAS_BUSQUEDA):
        h = ahora - dt.timedelta(hours=atras)
        for formato in FORMATOS_RUTA:
            prefijo = f'{PREFIJO}/{h.strftime(formato)}/'
            contenido = con_reintento(lambda: pedir(prefijo))
            claves = [e.text for e in ET.fromstring(contenido).findall('s3:Contents/s3:Key', ns)
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


def ejes(ds):
    """Ejes de escaneo en radianes. Este producto trae x e y como ÍNDICES 0..N-1 (sin unidades), no como ángulos:
    en ese caso se reconstruyen con la rejilla fija estándar de GOES-R. (Antes se usaban tal cual y TODOS los puntos
    caían en el píxel 0,0 = espacio vacío: la lluvia por satélite nunca midió nada.)"""
    xs, ys = ds['x'].values.astype('float64'), ds['y'].values.astype('float64')
    if max(np.abs(xs).max(), np.abs(ys).max()) <= 1.0:
        return xs, ys                                          # ya vienen en radianes
    paso = 2 * OFF_REJILLA / (xs.size - 1)
    return -OFF_REJILLA + paso * np.arange(xs.size), OFF_REJILLA - paso * np.arange(ys.size)


def verificar_mapeo(ds, LAT, LON, dentro, y0, y1, x0, x1, jy, jx):
    """Comprueba con las Latitude/Longitude del propio archivo que cada punto cayó en el píxel correcto.
    Si el error es grande ABORTA: mejor no guardar nada que guardar la lluvia de otro lugar. Devuelve el peor error (grados)."""
    if 'Latitude' not in ds.variables or 'Longitude' not in ds.variables:
        log.warning('El archivo no trae Latitude/Longitude: no se pudo verificar el mapeo')
        return None

    def rect(nombre):
        v = ds[nombre]
        d0, d1 = v.dims[-2:]
        a = v.isel({d0: slice(int(y0), int(y1) + 1), d1: slice(int(x0), int(x1) + 1)}).values.astype('float64')
        return (a.T if d0 == 'x' else a)[jy, jx]

    err = np.maximum(np.abs(rect('Latitude') - LAT), np.abs(rect('Longitude') - LON))[dentro]
    err = err[np.isfinite(err)]
    if err.size == 0:
        raise RuntimeError('Ningún punto tiene Latitude/Longitude válida: el mapeo a píxeles está mal')
    peor = float(err.max())
    if peor > TOLERANCIA_MAPEO_GRADOS:
        raise RuntimeError(f'Mapeo inválido: el píxel elegido está hasta {peor:.3f}° del punto pedido (máximo {TOLERANCIA_MAPEO_GRADOS}°)')
    return round(peor, 4)


# ---------- 3b. Diagnóstico de calidad (solo informa; no afecta lo que se guarda) ----------
def atributos_calidad(ds, nombre):
    """Lo que el propio archivo dice de una variable de calidad (significado de sus códigos). Texto recortado."""
    a = ds[nombre].attrs
    return {k: str(a[k])[:400] for k in ('long_name', 'flag_values', 'flag_masks', 'flag_meanings', 'comment') if k in a}


def desglose(codigos, valores, dentro):
    """Por cada código de calidad: cuántos puntos, cuántos con lluvia >= UMBRAL_LOG_MM y el máximo.
    Solo cuenta puntos dentro de la imagen y con valor válido (>= 0); el relleno -999 no entra."""
    ok = dentro & np.isfinite(valores) & (valores >= 0) & np.isfinite(codigos)
    filas = []
    for c in np.unique(codigos[ok]):
        v = valores[ok & (codigos == c)]
        filas.append({'codigo': int(c), 'n': int(v.size), 'con_lluvia': int((v >= UMBRAL_LOG_MM).sum()), 'max': round(float(v.max()), 1)})
    return sorted(filas, key=lambda f: -f['n'])


def registrar_diagnostico(diag):
    """Escribe el diagnóstico en el log: atributos de DQF/PQI y el desglose por código."""
    if not diag:
        return
    if diag.get('error'):
        log.warning('Diagnóstico de calidad incompleto: %s', diag['error'])
    for nombre, attrs in (diag.get('attrs') or {}).items():
        log.info('Atributos de %s según el archivo: %s', nombre, json.dumps(attrs, ensure_ascii=False))
    for clave, etiqueta in (('dqf', 'DQF'), ('pqi', 'PQI')):
        for f in (diag.get(clave) or [])[:TOP_CODIGOS_LOG]:
            log.info('%s %s: %d puntos | %d con lluvia >= %.0f mm/h | máx %.1f mm/h', etiqueta, f['codigo'], f['n'], f['con_lluvia'], UMBRAL_LOG_MM, f['max'])


def muestrear(ds):
    """Devuelve (rejilla, estadísticas). Fila 0 = latitud más al sur, columna 0 = longitud más al oeste.
    En 'mm': número = lluvia (0.0 = hay dato y no llueve); None = SIN INFORMACIÓN."""
    nombre, nom_dqf, p = variable_lluvia(ds), variable_calidad(ds), parametros(ds)
    xs, ys = ejes(ds)
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
    error_mapeo = verificar_mapeo(ds, LAT, LON, dentro, y0, y1, x0, x1, jy, jx)   # aborta si el mapeo está mal

    # Clasificación de cada punto: ¿hay dato fiable o no? (reglas fijas, ver arriba)
    nan = ~np.isfinite(valores)
    absurdo = ~nan & ((valores < 0) | (valores >= MM_MAX_VALIDO))
    if calidad is not None:
        cod = np.where(np.isfinite(calidad), calidad, 0).astype(int)
        dqf_malo = ~np.isfinite(calidad) | ((cod & BIT_RECUPERACION_MALA) != 0)    # sin código, o bit 0 encendido
    else:
        dqf_malo = np.zeros(valores.shape, dtype=bool)
    sin_dato = ~dentro | dqf_malo | absurdo | nan                   # un NaN nunca se convierte en 0.0
    mm = np.where(sin_dato, np.nan, valores)                        # NaN aquí solo marca "sin dato" (luego será null en el JSON)

    # Diagnóstico de calidad: nunca debe romper la ejecución, por eso va protegido
    diag = {}
    try:
        diag['attrs'] = {n: atributos_calidad(ds, n) for n in (nom_dqf, 'PQI') if n and n in ds.data_vars}
        if calidad is not None:
            diag['dqf'] = desglose(calidad, valores, dentro)
        if 'PQI' in ds.data_vars:
            diag['pqi'] = desglose(leer('PQI'), valores, dentro)
    except Exception as e:                                          # el diagnóstico es opcional
        diag['error'] = f'{type(e).__name__}: {str(e)[:150]}'

    con_dato = int((~sin_dato).sum())
    with np.errstate(invalid='ignore'):
        n_lluvia = int((mm >= UMBRAL_LOG_MM).sum())                 # los NaN comparan como False
    stats = {
        'puntos': int(mm.size), 'con_dato': con_dato, 'fuera_de_vista': int((~dentro).sum()),
        'dqf_malo': int((dentro & dqf_malo).sum()), 'absurdo': int((dentro & absurdo).sum()), 'nan': int((dentro & nan).sum()),
        'dqf_hist': {str(k): int(v) for k, v in zip(*np.unique(calidad[dentro], return_counts=True))} if calidad is not None else None,
        'dqf_var': nom_dqf, 'error_mapeo': error_mapeo,
        'n_lluvia': n_lluvia, 'max_mm': float(np.nanmax(mm)) if con_dato else 0.0, 'diag': diag,
    }
    rejilla = {
        'variable': nombre, 'dqf': nom_dqf,
        'lat0': round(float(lats[0]), 4), 'lon0': round(float(lons[0]), 4), 'paso': PASO,
        'filas': int(lats.size), 'cols': int(lons.size),
        'cobertura': round(con_dato / stats['puntos'], 3),             # fracción de puntos con información
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
    cuerpo = {
        'query': 'INSERT INTO lluvia_sat (id, escaneo, rejilla) VALUES (1, $1::timestamptz, $2::jsonb) '
                 'ON CONFLICT (id) DO UPDATE SET escaneo = EXCLUDED.escaneo, rejilla = EXCLUDED.rejilla, actualizado_en = now()',
        'params': [escaneo.isoformat(), json.dumps(rejilla, allow_nan=False)],   # allow_nan=False: nunca sale un NaN inválido
    }
    r = con_reintento(lambda: requests.post(
        f'https://{host}/sql',
        headers={'Content-Type': 'application/json', 'Neon-Connection-String': url},
        json=cuerpo, timeout=30))
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

    pct_lluvia = 100 * st['n_lluvia'] / st['con_dato'] if st['con_dato'] else 0
    log.info('Variable de lluvia: %s | calidad: %s | error máximo de mapeo vs Latitude/Longitude del archivo: %s°',
             rejilla['variable'], st['dqf_var'] or 'NO EXISTE', st['error_mapeo'])
    log.info('Puntos: %d | con dato: %d (%.0f%%) | sin dato: fuera de vista=%d, calidad mala=%d, valor absurdo=%d | NaN=%d',
             st['puntos'], st['con_dato'], rejilla['cobertura'] * 100, st['fuera_de_vista'], st['dqf_malo'], st['absurdo'], st['nan'])
    log.info('Códigos DQF en el área: %s | se descarta si el bit 0 está encendido', st['dqf_hist'])
    log.info('Puntos con lluvia >= %.0f mm/h: %d de %d con dato (%.0f%%) | máximo: %.1f mm/h',
             UMBRAL_LOG_MM, st['n_lluvia'], st['con_dato'], pct_lluvia, st['max_mm'])
    registrar_diagnostico(st['diag'])
    guardar(escaneo, rejilla)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('Falló la ejecución')       # imprime la traza completa con la línea exacta
        sys.exit(1)
