#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: DIAGNÓSTICO v3 (de una sola vez, no guarda nada en Neon)
#
# Hipótesis a comprobar: los ejes x e y del archivo NO vienen en radianes (son 0..5423), así que nuestra
# fórmula mandaba TODOS los puntos al píxel (0,0), una esquina vacía del satélite (DQF 3 y -999).
# Se comprueba de tres formas independientes:
#   1. Se imprimen los ejes x e y tal como vienen.
#   2. Se busca el píxel de Guayaquil usando las Latitude/Longitude 2D del propio archivo (no depende de fórmulas).
#   3. Se calcula el píxel con las constantes estándar de la rejilla GOES-R 2 km y se compara con (2).
# Reutiliza listar_ultimo(), abrir(), parametros() y proyectar() de sat_lluvia.py (misma carpeta scripts/).
# =====================================================
import logging
import sys
import warnings

import numpy as np

import sat_lluvia as s

log = logging.getLogger('diagnostico')
warnings.simplefilter('ignore')
# Constantes estándar de la rejilla fija GOES-R de 2 km (ángulo de escaneo en radianes por píxel)
ESC, OFF = 5.6e-5, 0.151844
GYE = (-2.19, -79.89)          # centro de Guayaquil


def bloque(ds, nombre, y0, y1, x0, x1):
    """Rectángulo de una variable 2D como matriz [fila, columna]."""
    v = ds[nombre]
    d0, d1 = v.dims[-2:]
    a = v.isel({d0: slice(int(y0), int(y1) + 1), d1: slice(int(x0), int(x1) + 1)}).values.astype('float64')
    return a.T if d0 == 'x' else a


def por_codigo(etiqueta, dq, rr):
    """Histograma de DQF y, por código, qué valores de lluvia trae."""
    cod, n = np.unique(dq[np.isfinite(dq)], return_counts=True)
    log.info('%s | histograma DQF: %s', etiqueta, {str(c): int(k) for c, k in zip(cod, n)})
    for c in cod:
        fin = rr[(dq == c) & np.isfinite(rr)]
        log.info('%s | DQF=%s | píxeles=%d | mínimo=%s | máximo=%s | con lluvia>=2: %d', etiqueta, c, fin.size,
                 fin.min() if fin.size else None, fin.max() if fin.size else None, int((fin >= 2).sum()))


def main():
    clave = s.listar_ultimo()
    log.info('Archivo: %s', clave)
    with s.abrir(clave) as ds:
        # ---- 1. Los ejes tal como vienen ----
        log.info('Coordenadas del dataset: %s | goes_imager_projection presente: %s', list(ds.coords), 'goes_imager_projection' in ds.variables)
        for n in ('x', 'y'):
            v = ds[n].values
            log.info('Eje %s: tipo=%s | primeros=%s | últimos=%s | atributos=%s', n, v.dtype, v[:3], v[-3:], dict(ds[n].attrs))
        N = ds.sizes['x']

        # ---- 2. Píxel de Guayaquil según las Latitude/Longitude del archivo (búsqueda gruesa, 1 de cada 8) ----
        k = 8
        la_n, lo_n = ds['Latitude'], ds['Longitude']
        d0, d1 = la_n.dims
        la = la_n.isel({d0: slice(None, None, k), d1: slice(None, None, k)}).values.astype('float64')
        lo = lo_n.isel({d0: slice(None, None, k), d1: slice(None, None, k)}).values.astype('float64')
        dist = (la - GYE[0]) ** 2 + ((lo - GYE[1]) * np.cos(np.radians(GYE[0]))) ** 2
        dist = np.where(np.isfinite(dist), dist, np.inf)
        f, c = np.unravel_index(dist.argmin(), dist.shape)
        log.info('REFERENCIA del archivo: Guayaquil está cerca del píxel fila=%d columna=%d (±%d) | su lat/lon=%.3f, %.3f',
                 f * k, c * k, k, la[f, c], lo[f, c])

        # ---- 3. Mismo punto con las constantes estándar y con nuestra fórmula ----
        xs_std = -OFF + ESC * np.arange(N)
        ys_std = OFF - ESC * np.arange(N)
        p = s.parametros(ds)
        x, y, vis = s.proyectar(np.array(GYE[0]), np.array(GYE[1]), p)
        log.info('Ángulos de escaneo de Guayaquil: x=%.5f y=%.5f rad | visible=%s', x, y, bool(vis))
        ix_std, iy_std = int(s.indice_en_eje(xs_std, np.array([x]))[0]), int(s.indice_en_eje(ys_std, np.array([y]))[0])
        ix_arch, iy_arch = int(s.indice_en_eje(ds['x'].values.astype('float64'), np.array([x]))[0]), int(s.indice_en_eje(ds['y'].values.astype('float64'), np.array([y]))[0])
        log.info('PÍXEL con constantes estándar: fila=%d columna=%d | PÍXEL con los ejes del archivo (lo que usa hoy el script): fila=%d columna=%d',
                 iy_std, ix_std, iy_arch, ix_arch)

        # ---- 4. Prueba de fuego: error de latitud/longitud en nuestros 380 puntos con las constantes estándar ----
        lats = np.arange(s.LAT_MIN, s.LAT_MAX + s.PASO / 2, s.PASO)
        lons = np.arange(s.LON_MIN, s.LON_MAX + s.PASO / 2, s.PASO)
        LAT, LON = np.meshgrid(lats, lons, indexing='ij')
        with np.errstate(invalid='ignore', divide='ignore'):
            xx, yy, vv = s.proyectar(LAT, LON, p)
            ix, iy = s.indice_en_eje(xs_std, xx), s.indice_en_eje(ys_std, yy)
        ok = vv & (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
        y0, y1, x0, x1 = iy[ok].min(), iy[ok].max(), ix[ok].min(), ix[ok].max()
        jy, jx = np.clip(iy, y0, y1) - y0, np.clip(ix, x0, x1) - x0
        el = np.abs(bloque(ds, 'Latitude', y0, y1, x0, x1)[jy, jx] - LAT)
        eo = np.abs(bloque(ds, 'Longitude', y0, y1, x0, x1)[jy, jx] - LON)
        log.info('MAPEO con constantes estándar: puntos con coordenada válida=%d de %d | error lat medio=%.4f° máx=%.4f° | error lon medio=%.4f° máx=%.4f° (bien si < 0.05°)',
                 int(np.isfinite(el).sum()), el.size, np.nanmean(el), np.nanmax(el), np.nanmean(eo), np.nanmax(eo))
        por_codigo('NUESTRA ZONA con mapeo estándar', bloque(ds, 'DQF', y0, y1, x0, x1)[jy, jx], bloque(ds, 'RRQPE', y0, y1, x0, x1)[jy, jx])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('Falló el diagnóstico')
        sys.exit(1)
