#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: DIAGNÓSTICO v2 (de una sola vez, no guarda nada en Neon)
#
# Pregunta que responde: ¿por qué TODOS los puntos de nuestra zona salen con DQF 3 y valor -999?
#   a) ¿Nuestra proyección apunta al píxel equivocado?  -> se compara con las Latitude/Longitude del propio archivo.
#   b) ¿O de verdad no hay estimación sobre la zona?     -> se mira una ventana grande alrededor, a resolución completa.
# Reutiliza listar_ultimo(), abrir(), proyectar() e indice_en_eje() de sat_lluvia.py (misma carpeta scripts/).
# =====================================================
import logging
import sys

import numpy as np

import sat_lluvia as s

log = logging.getLogger('diagnostico')


def bloque(ds, nombre, y0, y1, x0, x1):
    """Lee un rectángulo de una variable 2D como matriz [fila(y), columna(x)], sea cual sea el orden de sus ejes."""
    v = ds[nombre]
    d0, d1 = v.dims[-2:]
    a = v.isel({d0: slice(int(y0), int(y1) + 1), d1: slice(int(x0), int(x1) + 1)}).values.astype('float64')
    return a.T if d0 == 'x' else a


def indices(ds, lats, lons):
    """Píxel (iy, ix) de cada punto lat/lon con nuestra fórmula."""
    p = s.parametros(ds)
    xs, ys = ds['x'].values.astype('float64'), ds['y'].values.astype('float64')
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    with np.errstate(invalid='ignore', divide='ignore'):
        x, y, vis = s.proyectar(LAT, LON, p)
        ix, iy = s.indice_en_eje(xs, x), s.indice_en_eje(ys, y)
    dentro = vis & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    return LAT, LON, ix, iy, dentro


def por_codigo(etiqueta, dq, rr):
    """Histograma de DQF y, por código, qué valores de lluvia trae."""
    cod, n = np.unique(dq[np.isfinite(dq)], return_counts=True)
    log.info('%s | histograma DQF: %s', etiqueta, {str(c): int(k) for c, k in zip(cod, n)})
    for c in cod:
        v = rr[dq == c]
        fin = v[np.isfinite(v)]
        log.info('%s | DQF=%s | píxeles=%d | mínimo=%s | máximo=%s | con lluvia>=2: %d', etiqueta, c, v.size,
                 fin.min() if fin.size else None, fin.max() if fin.size else None, int((fin >= 2).sum()))


def main():
    clave = s.listar_ultimo()
    log.info('Archivo: %s', clave)
    with s.abrir(clave) as ds:
        log.info('DQF significado COMPLETO: %s', ds['DQF'].attrs.get('long_name'))
        log.info('Dimensiones de RRQPE: %s | Latitude: %s | Longitude: %s', ds['RRQPE'].dims,
                 ds['Latitude'].dims if 'Latitude' in ds else None, ds['Longitude'].dims if 'Longitude' in ds else None)

        # ---- A) Nuestra rejilla: ¿el píxel elegido está donde creemos? ----
        lats = np.arange(s.LAT_MIN, s.LAT_MAX + s.PASO / 2, s.PASO)
        lons = np.arange(s.LON_MIN, s.LON_MAX + s.PASO / 2, s.PASO)
        LAT, LON, ix, iy, dentro = indices(ds, lats, lons)
        y0, y1, x0, x1 = iy[dentro].min(), iy[dentro].max(), ix[dentro].min(), ix[dentro].max()
        jy, jx = np.clip(iy, y0, y1) - y0, np.clip(ix, x0, x1) - x0
        try:
            la = bloque(ds, 'Latitude', y0, y1, x0, x1)[jy, jx]
            lo = bloque(ds, 'Longitude', y0, y1, x0, x1)[jy, jx]
            el, eo = np.abs(la - LAT), np.abs(lo - LON)
            log.info('MAPEO: error medio lat=%.4f° lon=%.4f° | error máximo lat=%.4f° lon=%.4f° (más de 0.05° = proyección mal)',
                     np.nanmean(el), np.nanmean(eo), np.nanmax(el), np.nanmax(eo))
        except Exception as e:
            log.warning('No se pudo comparar con Latitude/Longitude del archivo: %s: %s', type(e).__name__, str(e)[:150])
        dq = bloque(ds, 'DQF', y0, y1, x0, x1)[jy, jx]
        rr = bloque(ds, 'RRQPE', y0, y1, x0, x1)[jy, jx]
        por_codigo('NUESTRA ZONA (380 puntos)', dq, rr)
        for n in ('solar_zenith_angle', 'retrieval_local_zenith_angle'):
            try:
                a = bloque(ds, n, y0, y1, x0, x1)[jy, jx]
                log.info('%s en nuestra zona: mínimo=%.1f máximo=%.1f', n, np.nanmin(a), np.nanmax(a))
            except Exception as e:
                log.info('%s: no se pudo leer (%s)', n, type(e).__name__)

        # ---- B) Ventana grande a resolución completa: costa de Ecuador y mar cercano ----
        for nombre, (la0, la1, lo0, lo1) in {'VENTANA 2°x2° alrededor de Guayaquil': (-3.2, -1.2, -80.9, -78.9),
                                              'VENTANA 8°x10° (Ecuador, Perú norte, Pacífico)': (-6.0, 2.0, -85.0, -75.0)}.items():
            g1, g2 = np.arange(la0, la1, 0.05), np.arange(lo0, lo1, 0.05)
            _, _, bx, by, ok = indices(ds, g1, g2)
            if not ok.any():
                log.info('%s: fuera de vista', nombre)
                continue
            a, b, c, d = by[ok].min(), by[ok].max(), bx[ok].min(), bx[ok].max()
            por_codigo(nombre, bloque(ds, 'DQF', a, b, c, d), bloque(ds, 'RRQPE', a, b, c, d))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('Falló el diagnóstico')
        sys.exit(1)
