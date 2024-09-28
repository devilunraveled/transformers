import re
from .data_config import DataConfig

def loadFile(path : str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def saveFile(path : str, data : str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def removeSpecialCharacters(text : str, regex : str = DataConfig['regex']) -> str:
    """
        Remove special characters from a string, 
        only keeping alphanumeric characters
        and spaces. Also, converts the entire 
        text to lowecase.
    """
    return re.sub(regex, '', text.lower())
