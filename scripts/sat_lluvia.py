#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: lluvia estimada por satélite (NOAA Enterprise Rain Rate, GOES-19)
#
# Qué hace, en orden:
#   1. Busca en el bucket público de NOAA el archivo más reciente (uno cada 10 min, ~68 MB).
#   2. Lo abre SIN descargarlo entero (solo lee los trozos que necesita).
#   3. Toma la lluvia (mm/h) en una rejilla de puntos sobre Guayaquil / Samborondón / Durán.
#   4. Guarda la rejilla en la base Neon (tabla lluvia_sat) para que la app la lea vía /api/lluvia-sat.
#
# Variable de entorno: DATABASE_URL (la misma de Cloudflare). Si no existe, solo muestra el resultado.
# Uso de prueba:  python scripts/sat_lluvia.py
# =====================================================
import os
import re
import sys
import json
import math
import datetime as dt
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import numpy as np
import requests
import xarray as xr

# ---------- Configuración ----------
BUCKET = 'https://noaa-enterprise-rainrate-pds.s3.amazonaws.com'
PREFIJO = 'G19/ABI-L2-FLDK-RainfallRate'      # GOES-19, disco completo
# Área que se muestrea (cubre Guayaquil, Samborondón, Durán, Pascuales y Vía a la Costa)
LAT_MIN, LAT_MAX = -2.36, -2.00
LON_MIN, LON_MAX = -80.04, -79.76
PASO = 0.02                                   # separación entre puntos (grados, ~2.2 km)
# Valores por defecto de la órbita de GOES-Este (se usan solo si el archivo no trae su proyección)
PROY_DEFECTO = {'req': 6378137.0, 'rpol': 6356752.31414, 'h': 35786023.0, 'lam0': -75.2}


# ---------- 1. Buscar el archivo más reciente ----------
def listar_ultimo():
    """Devuelve la clave (ruta) del archivo más nuevo; mira la hora UTC actual y hasta 3 horas atrás."""
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    ahora = dt.datetime.now(dt.timezone.utc)
    for atras in range(4):
        h = ahora - dt.timedelta(hours=atras)
        prefijo = f'{PREFIJO}/{h:%Y/%m/%d/%H}/'            # año/mes/día/hora UTC
        r = requests.get(BUCKET + '/', params={'list-type': '2', 'prefix': prefijo}, timeout=30)
        r.raise_for_status()
        claves = [e.text for e in ET.fromstring(r.content).findall('s3:Contents/s3:Key', ns)
                  if e.text and e.text.endswith('.nc')]
        if claves:
            return sorted(claves)[-1]                       # el nombre incluye la hora: el último es el más nuevo
    raise RuntimeError('No hay archivos en las últimas 4 horas')


def hora_escaneo(clave):
    """Lee la hora de inicio del escaneo desde el nombre (…_sAAAAMMDDhhmmssd_…)."""
    m = re.search(r'_s(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\d_', clave)
    if not m:
        raise RuntimeError('No se pudo leer la hora en el nombre: ' + clave)
    return dt.datetime(*map(int, m.groups()), tzinfo=dt.timezone.utc)


# ---------- 2. Abrir el archivo ----------
def abrir(clave):
    """Abre el NetCDF leyendo por partes (rápido). Si eso falla, lo descarga completo a /tmp."""
    url = f'{BUCKET}/{clave}'
    try:
        import fsspec
        f = fsspec.open(url, 'rb', block_size=4 * 1024 * 1024).open()
        return xr.open_dataset(f, engine='h5netcdf')
    except Exception as e:
        print('Lectura por partes falló (' + str(e)[:120] + '); se descarga el archivo completo...')
        ruta = '/tmp/lluvia.nc'
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(ruta, 'wb') as out:
                for trozo in r.iter_content(1 << 20):
                    out.write(trozo)
        return xr.open_dataset(ruta, engine='h5netcdf')


# ---------- 3. Muestrear la lluvia ----------
def variable_lluvia(ds):
    """Encuentra la variable con la tasa de lluvia (normalmente 'RRQPE')."""
    for n in ('RRQPE', 'RainfallRate', 'rain_rate', 'RR'):
        if n in ds.data_vars:
            return n
    for n, v in ds.data_vars.items():                       # respaldo: primera variable con ejes x e y
        if 'x' in v.dims and 'y' in v.dims and not n.startswith('DQF'):
            return n
    raise RuntimeError('No encontré la variable de lluvia. Variables: ' + ', '.join(ds.data_vars))


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
    """Convierte latitud/longitud al ángulo de escaneo (x, y en radianes) del satélite (fórmulas del manual GOES-R)."""
    phi, lam, lam0 = math.radians(lat), math.radians(lon), math.radians(p['lam0'])
    e2 = (p['req'] ** 2 - p['rpol'] ** 2) / p['req'] ** 2
    phic = math.atan((p['rpol'] ** 2 / p['req'] ** 2) * math.tan(phi))
    rc = p['rpol'] / math.sqrt(1 - e2 * math.cos(phic) ** 2)
    sx = p['h'] - rc * math.cos(phic) * math.cos(lam - lam0)
    sy = -rc * math.cos(phic) * math.sin(lam - lam0)
    sz = rc * math.sin(phic)
    y = math.atan(sz / sx)
    x = math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz))
    return x, y


def muestrear(ds):
    """Devuelve la rejilla de lluvia: fila 0 = latitud más al sur, columna 0 = longitud más al oeste."""
    nombre = variable_lluvia(ds)
    p = parametros(ds)
    xs, ys = ds['x'].values, ds['y'].values
    lats = np.arange(LAT_MIN, LAT_MAX + PASO / 2, PASO)
    lons = np.arange(LON_MIN, LON_MAX + PASO / 2, PASO)

    # Píxel del archivo más cercano a cada punto de la rejilla
    ix = np.zeros((len(lats), len(lons)), dtype=int)
    iy = np.zeros_like(ix)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            x, y = proyectar(float(la), float(lo), p)
            ix[i, j] = int(np.abs(xs - x).argmin())
            iy[i, j] = int(np.abs(ys - y).argmin())

    # Se lee solo el rectángulo que contiene esos píxeles (pocos KB), no toda la imagen
    y0, y1, x0, x1 = iy.min(), iy.max(), ix.min(), ix.max()
    bloque = ds[nombre].isel(y=slice(int(y0), int(y1) + 1), x=slice(int(x0), int(x1) + 1)).transpose('y', 'x').values
    vals = bloque.astype('float64')[iy - y0, ix - x0]
    vals = np.where(np.isfinite(vals) & (vals > 0) & (vals < 300), vals, 0.0)   # vacíos y valores absurdos = 0

    return {
        'variable': nombre,
        'lat0': round(float(lats[0]), 4), 'lon0': round(float(lons[0]), 4), 'paso': PASO,
        'filas': len(lats), 'cols': len(lons),
        'mm': [[round(float(v), 1) for v in fila] for fila in vals],
    }


# ---------- 4. Guardar en Neon ----------
def guardar(escaneo, rejilla):
    """Guarda (reemplaza) la última rejilla usando la API SQL por HTTP de Neon, igual que mareas.js."""
    url = os.environ.get('DATABASE_URL')
    if not url:
        print('Sin DATABASE_URL: no se guarda (modo prueba).')
        return
    host = urlparse(url).hostname
    r = requests.post(
        f'https://{host}/sql',
        headers={'Content-Type': 'application/json', 'Neon-Connection-String': url},
        json={
            'query': 'INSERT INTO lluvia_sat (id, escaneo, rejilla) VALUES (1, $1::timestamptz, $2::jsonb) '
                     'ON CONFLICT (id) DO UPDATE SET escaneo = EXCLUDED.escaneo, rejilla = EXCLUDED.rejilla, actualizado_en = now()',
            'params': [escaneo.isoformat(), json.dumps(rejilla)],
        },
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f'Neon respondió {r.status_code}: {r.text[:300]}')
    print('Guardado en Neon.')


def main():
    clave = listar_ultimo()
    escaneo = hora_escaneo(clave)
    edad = (dt.datetime.now(dt.timezone.utc) - escaneo).total_seconds() / 60
    print(f'Archivo: {clave}\nEscaneo: {escaneo.isoformat()} (hace {edad:.0f} min)')

    ds = abrir(clave)
    print('Variables:', list(ds.data_vars), '| dimensiones:', dict(ds.sizes))   # sirve para depurar la primera vez
    rejilla = muestrear(ds)

    plano = np.array(rejilla['mm'])
    print(f"Variable usada: {rejilla['variable']} | puntos con lluvia >= 2 mm/h: {(plano >= 2).sum()} de {plano.size} | máximo: {plano.max():.1f} mm/h")
    guardar(escaneo, rejilla)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
