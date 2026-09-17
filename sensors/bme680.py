"""SAFEBAND AI - BME680 environmental sensor interface.

Desktop prototype mode is simulated. The interface exposes the core BME680
measurements needed by the Phase-1 IAQ model and the future ESP32-S3 driver.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict
import random

SENSOR_NAME="BME680"; SENSOR_TYPE="Environmental"; INTERFACE="I2C"
DEFAULT_TEMPERATURE=27.0; DEFAULT_HUMIDITY=58.0; DEFAULT_PRESSURE=1012.0
DEFAULT_GAS_RESISTANCE_OHMS=500000.0

@dataclass
class BME680Reading:
    temperature: float
    humidity: float
    pressure: float
    gas_resistance_ohms: float
    heat_stable: bool
    connected: bool
    simulated: bool=True

class BME680Sensor:
    def __init__(self,simulation:bool=True)->None:
        self.simulation=bool(simulation); self.connected=False
        self.temperature=DEFAULT_TEMPERATURE; self.humidity=DEFAULT_HUMIDITY
        self.pressure=DEFAULT_PRESSURE; self.gas_resistance_ohms=DEFAULT_GAS_RESISTANCE_OHMS
        self.heat_stable=True
    def connect(self)->bool:
        if self.simulation: self.connected=True; return True
        self.connected=False; return False
    def disconnect(self): self.connected=False
    def _ensure_connected(self):
        if not self.connected:self.connect()
    def read_temperature(self):
        self._ensure_connected()
        if self.simulation:self.temperature=max(-40,min(85,self.temperature+random.uniform(-.25,.25)))
        return round(self.temperature,2)
    def read_humidity(self):
        self._ensure_connected()
        if self.simulation:self.humidity=max(0,min(100,self.humidity+random.uniform(-.8,.8)))
        return round(self.humidity,2)
    def read_pressure(self):
        self._ensure_connected()
        if self.simulation:self.pressure=max(300,min(1100,self.pressure+random.uniform(-1,1)))
        return round(self.pressure,2)
    def read_gas_resistance(self):
        self._ensure_connected()
        if self.simulation:self.gas_resistance_ohms=max(1000,self.gas_resistance_ohms+random.uniform(-15000,15000))
        return round(self.gas_resistance_ohms,2)
    def read(self)->BME680Reading:
        self._ensure_connected()
        return BME680Reading(self.read_temperature(),self.read_humidity(),self.read_pressure(),
            self.read_gas_resistance(),self.heat_stable,self.connected,self.simulation)
    def read_dict(self)->Dict[str,Any]:
        r=self.read()
        return {"temperature":r.temperature,"humidity":r.humidity,"pressure":r.pressure,
                "gas_resistance_ohms":r.gas_resistance_ohms,"heat_stable":r.heat_stable,
                "bme680_connected":r.connected,"simulated":r.simulated}
    def get_status(self)->Dict[str,Any]:
        return {"sensor":SENSOR_NAME,"name":"Environmental Sensor","type":SENSOR_TYPE,"interface":INTERFACE,
                "connected":self.connected,"simulation":self.simulation,"temperature":round(self.temperature,2),
                "humidity":round(self.humidity,2),"pressure":round(self.pressure,2),
                "gas_resistance_ohms":round(self.gas_resistance_ohms,2),"heat_stable":self.heat_stable}
    def reset(self):
        self.temperature=DEFAULT_TEMPERATURE; self.humidity=DEFAULT_HUMIDITY
        self.pressure=DEFAULT_PRESSURE; self.gas_resistance_ohms=DEFAULT_GAS_RESISTANCE_OHMS; self.heat_stable=True

_bme680=BME680Sensor(simulation=True)
def initialize_bme680()->bool:return _bme680.connect()
def read_bme680()->Dict[str,Any]:return _bme680.read_dict()
def get_bme680_status()->Dict[str,Any]:return _bme680.get_status()
def reset_bme680()->None:_bme680.reset()
__all__=["BME680Reading","BME680Sensor","initialize_bme680","read_bme680","get_bme680_status","reset_bme680"]
