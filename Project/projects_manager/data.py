import requests
from fastapi.responses import JSONResponse

def IndexToDataType(index: int) -> str:
    available_data_types = requests.get("http://db_service:8002/enums/datatypes").json()
    print(available_data_types)
    if not available_data_types:
        raise Exception("Failed to fetch data types from DB service")

    if index < 0 or index >= len(available_data_types):
        raise IndexError(f"Data type index out of range {index}")
    return available_data_types[index]

def DataTypeToIndex(data_type: str) -> int:
    available_data_types = requests.get("http://db_service:8002/enums/datatypes").json()
    print(available_data_types)
    if not available_data_types:
        raise Exception("Failed to fetch data types from DB service")

    if data_type not in available_data_types:
        raise ValueError(f"Data type not found ({data_type}). Expected one of: {available_data_types}")
    return available_data_types.index(data_type)