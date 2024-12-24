import csv
from pathlib import Path
from typing import List, Dict


class CsvReader:
    """
    Helper class for reading and parsing CSV files with specific encoding and delimiter.

    This class handles the reading of CSV files while managing file encoding and proper resource cleanup.

    Examples:
        reader = CsvReader('data.csv')
        records = reader.read()
        print(f"Loaded {len(records)} records")

    Attributes:
        file_path (Path): Path to the CSV file to be read
    """

    def __init__(self, file_path: Path):
        """
        Initialize the CSV reader with a file path.

        :param file_path: Path to the CSV file
        :type file_path: Path
        """
        self.file_path = file_path

    def read(self) -> List[Dict]:
        """
        Read the CSV file and return its contents as a list of dictionaries.

        :return: List of dictionaries where each dictionary represents a row in the CSV
        :raises FileNotFoundError: If the specified file does not exist
        :raises csv.Error: If there are issues parsing the CSV file
        """
        with open(self.file_path, encoding="utf-8-sig") as file:
            csv_reader = csv.DictReader(file, delimiter=';')
            return list(csv_reader)
