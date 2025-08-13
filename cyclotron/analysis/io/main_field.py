from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

from cyclotron.analysis.parameters.constants import ANGLE_OF_ZERO_INDEX_IN_DEGREES as _88AZI
from cyclotron.analysis.model import FieldMetadata, MagneticField

_log = getLogger(__name__)

# Values unique to the input files
_ENTRIES_PER_LINE: int = 5
_ANGLE_OF_ZERO_INDEX_IN_DEGREES: float = 33
_THETA_DELTA: float = 3
_RADIUS_OF_ZERO_INDEX_IN_INCHES: float = 0.0
_RADIUS_DELTA: float = 1

@dataclass
class CurrentRange:
    """Range of currents that identify the field coefficients"""
    minimum_current: float
    maximum_current: float

    def contains(self, value: float) -> bool:
        return self.minimum_current < value <= self.maximum_current
    
    def __eq__(self, rhs) -> bool:
        return self.minimum_current == rhs.minimum_current and self.maximum_current == rhs.maximum_current

@dataclass
class FieldCoefficients:
    current_range: CurrentRange
    coefficients: np.ndarray          # IxJxK array where i: coeff, j: theta, k:r
    metadata: FieldMetadata

@dataclass(init=True)
class RawFieldMetadata:
    number_of_coefficients: int
    minimum_current: float
    maximum_current: float
    r_min: float
    r_idx_min: int
    r_idx_max: int
    theta_idx_min: int
    theta_idx_max: int

    @property
    def full_block_length(self) -> int:
        """Length between metadata for each current values"""
        return 2*(self.block_length +1)
    
    @property
    def block_length(self) -> int:
        """Length between metadata entries"""
        n_r_idx = self.r_idx_max - self.r_idx_min + 1
        return int((self.coefficient_length + 1) * n_r_idx)

    @property
    def coefficient_length(self) -> int:
        """Length between k values"""
        n_theta_idx = self.theta_idx_max - self.theta_idx_min + 1
        return int(self.number_of_coefficients * n_theta_idx/_ENTRIES_PER_LINE)

    @classmethod
    def from_file(cls, raw_metadata: str) -> 'RawFieldMetadata':
        split_metadata = raw_metadata.split()
        if len(split_metadata) != 8:
            raise RuntimeError(f'Incorrect metadata: {raw_metadata}')
        return RawFieldMetadata(
            number_of_coefficients=int(split_metadata[0]),
            minimum_current=float(split_metadata[1]),
            maximum_current=float(split_metadata[2]),
            r_min=float(split_metadata[3]),
            theta_idx_min=int(split_metadata[4]),
            theta_idx_max=int(split_metadata[5]),
            r_idx_min=int(split_metadata[6]),
            r_idx_max=int(split_metadata[7])
        )

@dataclass
class FieldRawData:
    metadata: RawFieldMetadata
    coefficient_lines: List[str]

def _read_field_data(field_file: Path) -> List[FieldCoefficients]:
    # Convert field data in field_file to numpy and save to output
    _log.info(f'Converting field data at {field_file}')
    raw_field_data = _read_field_input(field_file)
    _log.debug(f'Found {len(raw_field_data)} blocks of raw data')
    field_data = _convert_raw_data(raw_field_data)
    _log.info(f'Processed {len(field_data)} current ranges')
    return field_data

def _convert_raw_data(raw_data: List[FieldRawData]) -> List[FieldCoefficients]:
    field_data: List[FieldCoefficients] = []
    for raw_fields in raw_data:
        current_range = CurrentRange(raw_fields.metadata.minimum_current, raw_fields.metadata.maximum_current)
        exists = False
        coefficients = _raw_data_to_numpy(raw_fields)
        for i, existing_fields in enumerate(field_data):
            if existing_fields.current_range == current_range:
                field_data[i].coefficients = np.add(field_data[i].coefficients, coefficients)
                exists = True
        if not exists:
            field_data.append(FieldCoefficients(current_range, 
                                                coefficients,
                                                metadata=FieldMetadata(r_min=_RADIUS_OF_ZERO_INDEX_IN_INCHES, 
                                                              delta_r=_RADIUS_DELTA, 
                                                              theta_min=_88AZI, 
                                                              delta_theta=_THETA_DELTA)))
                                            
    return field_data 

def _raw_data_to_numpy(raw_data: FieldRawData) -> np.ndarray:
    j_index_offset = int((_88AZI - _ANGLE_OF_ZERO_INDEX_IN_DEGREES)/_THETA_DELTA)
    if j_index_offset < 0:
        _log.error('j_index_offset is negative, b-field array may not fill correctly')
    md = raw_data.metadata
    # One matrix for each coefficient
    i_dim = md.number_of_coefficients
    # Rows are theta indexes
    j_dim = md.theta_idx_max - md.theta_idx_min + 1
    # Columns are r indexes (which are also values)
    k_dim = md.r_idx_max - md.r_idx_min + 1

    array = np.zeros((i_dim, j_dim, 2*k_dim))

    for k in range(k_dim):
        # Offset if this is for a larger range
        k_idx = int(k + md.r_min)
        start = int(k * (md.coefficient_length + 1) + 1)
        end = int((k + 1) * (md.coefficient_length + 1))
        k_data = [float(t) for v in raw_data.coefficient_lines[start:end] for t in v.split()]
        # Fill array. Data starts at 33 degrees, but for the 88 it should line up index
        # 0 with 45 degrees, so shift the array by 4
        for i in range(i_dim):
            for j in [(v + j_index_offset) % j_dim for v in range(j_dim)]:
                array[i][j][k_idx] = k_data[j*i_dim + i]
    return array
        

    
def _read_field_input(path: Path) -> List[FieldRawData]:
    _log.debug(f'Reading field input file: {path}')
    fields: List[FieldRawData] = []
    with open(path, 'r') as f:
        while f:
            # Process full metadata blocks
            # Skip header
            try:
                next(f)
            except StopIteration:
                break
            field_metadata = RawFieldMetadata.from_file(f.readline())
            coefficient_lines = [next(f) for _ in range(field_metadata.block_length)]
            fields.append(FieldRawData(field_metadata, coefficient_lines))
    return fields

def build_iron_field_from_file(current: float, data_path: Path) -> MagneticField:
    """Reads the file at data_path for all field-data, and determines
    the correct fields using the passed current. Returning a
    MagneticField object representing the "iron field"

    Args:
        current (float): main coil current
        data_path (Path): path to the magnetic field datafile

    Returns:
        MagneticField: the field corresponding to the passed current
    """
    return _calculate_fields(current, _read_field_data(data_path))


def _calculate_fields(current: float, 
                     all_coefficients: List[FieldCoefficients]) -> MagneticField:
    """Convert all possible field coefficients into a MagneticField object"""
    for c in all_coefficients:
        current_range, coefficients = c.current_range, c.coefficients
        if current_range.contains(current):
            n_coefficients = np.shape(coefficients)[0]
            currents = np.array(list(current**(p) for p in range(n_coefficients)))
            return MagneticField(values=np.einsum('i,ijk', currents, coefficients),
                                 metadata=c.metadata)
    _log.error(f'No appropriate current range for input current {current}')
    raise RuntimeError
