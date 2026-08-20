"""Módulo de Sensores."""
from .sensor_base import (
    Sensor,
    SensorDirecaoObjetivo,
    SensorVizinhanca,
    SensorDistancia,
    SensorIluminacao,
    SensorPosicao,
    SensorComposto,
    criar_sensor_farol_completo,
    criar_sensor_farol_limitado,
    criar_sensor_labirinto
)

__all__ = [
    'Sensor',
    'SensorDirecaoObjetivo',
    'SensorVizinhanca',
    'SensorDistancia',
    'SensorIluminacao',
    'SensorPosicao',
    'SensorComposto',
    'criar_sensor_farol_completo',
    'criar_sensor_farol_limitado',
    'criar_sensor_labirinto'
]
