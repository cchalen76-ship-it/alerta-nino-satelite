#!/usr/bin/env python3
# =====================================================
# AlertA-Niño: DIAGNÓSTICO de una sola vez (no guarda nada en Neon)
#
# Objetivo: entender qué significan los códigos DQF y los valores de relleno de RRQPE en este producto,
# mirando TODO el disco del satélite y no solo nuestra zona (donde hoy no hay nubes con lluvia).
# Reutiliza listar_ultimo() y abrir() de sat_lluvia.py (deben estar en la misma carpeta scripts/).
# Uso:  python scripts/diagnostico_dqf.py
# =====================================================
import logging
import sys

import numpy as np

import sat_lluvia as s

log = logging.getLogger('diagnostico')
PASO = 20                      # se mira 1 de cada 20 píxeles en cada eje (~73 mil puntos de todo el disco)


def atributos(ds, nombre):
    """Muestra los metadatos de una variable: aquí NOAA suele documentar el significado de los códigos."""
    if nombre not in ds:
        log.info('%s: no existe en el archivo', nombre)
        return
    v = ds[nombre]
    log.info('%s | tipo=%s | atributos=%s', nombre, v.dtype, {k: str(a)[:300] for k, a in v.attrs.items()})
    log.info('%s | codificación (relleno original)=%s', nombre, {k: str(c)[:80] for k, c in v.encoding.items() if k in ('_FillValue', 'dtype', 'scale_factor', 'add_offset')})


def main():
    clave = s.listar_ultimo()
    log.info('Archivo: %s', clave)
    with s.abrir(clave) as ds:
        for n in ('RRQPE', 'DQF', 'PQI', 'quality_information'):
            atributos(ds, n)
        for n in ('TotalRetrievals', 'TotalRainArea', 'NumQAVals'):
            if n in ds:
                log.info('%s = %s', n, ds[n].values)

        recorte = dict(y=slice(None, None, PASO), x=slice(None, None, PASO))
        dq = ds['DQF'].isel(**recorte).values.astype('float64')
        rr = ds['RRQPE'].isel(**recorte).values.astype('float64')

    log.info('Píxeles muestreados de todo el disco: %d', dq.size)
    codigos, cuentas = np.unique(dq[np.isfinite(dq)], return_counts=True)
    log.info('Histograma DQF (todo el disco): %s', {str(c): int(n) for c, n in zip(codigos, cuentas)})

    # Para cada código DQF: ¿qué valores de lluvia trae? Así se ve cuál es el relleno y cuál es lluvia real
    for c in codigos:
        v = rr[dq == c]
        fin = v[np.isfinite(v)]
        unicos = np.unique(fin)
        log.info('DQF=%s | píxeles=%d | NaN=%d | mínimo=%s | máximo=%s | valores distintos=%d %s',
                 c, v.size, int(np.isnan(v).sum()), fin.min() if fin.size else None, fin.max() if fin.size else None,
                 unicos.size, list(unicos[:6]) if unicos.size <= 6 else '(muchos)')

    util = rr[np.isfinite(rr) & (rr >= 0) & (rr < s.MM_MAX_VALIDO)]
    log.info('Píxeles con lluvia en rango válido [0, %.0f): %d | con lluvia > 0: %d | con lluvia >= 2 mm/h: %d | máximo: %.1f',
             s.MM_MAX_VALIDO, util.size, int((util > 0).sum()), int((util >= 2).sum()), util.max() if util.size else 0.0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('Falló el diagnóstico')
        sys.exit(1)
