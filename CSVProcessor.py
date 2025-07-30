import csv

class CSVProcessor:
    def __init__(self, filepath, encoding='utf-8', has_header=True):
        """
        filepath: CSV 文件路径
        encoding: 文件编码
        has_header: 文件是否有表头（默认 True）
        fieldnames: 如果没有表头，需要手动传入列名列表
        """
        self.filepath = filepath
        self.data = []
        self.header_index = {}
        self._load_csv(encoding, has_header)

    def _load_csv(self, encoding, has_header):
        with open(self.filepath, mode='r', encoding=encoding) as f:
            reader = csv.reader(f)
            if has_header:
                self.headers = next(reader)
                self.header_index = {name: idx for idx, name in enumerate(self.headers)}
                self.data = [row for row in reader]
            else:
                # If no header, generate default headers as col0, col1, ...
                first_row = next(reader)
                num_cols = len(first_row)
                self.headers = [f"col{i}" for i in range(num_cols)]
                self.header_index = {name: idx for idx, name in enumerate(self.headers)}
                self.data = [first_row] + [row for row in reader]
    
    def get_value(self, row_index, column_name):
        """
        获取指定行和列的值
        """
        if column_name not in self.header_index:
            raise KeyError(f"列名 '{column_name}' 不存在！")
        column_index = self.header_index[column_name]
        return self.get_value_by_index(row_index, column_index)
    
    def get_value_by_index(self, row_index, column_index):
        """
        获取指定行和列索引的值
        """
        if row_index < 0 or row_index >= len(self.data):
            raise IndexError(f"行索引 {row_index} 超出范围！")
        if column_index < 0 or column_index >= len(self.headers):
            raise IndexError(f"列索引 {column_index} 超出范围！")
        return self.data[row_index][column_index]
    
    def has_value_in_column(self, column_name, value):
        """
        判断指定列名中是否存在某个值
        """
        if column_name not in self.header_index:
            raise KeyError(f"列名 '{column_name}' 不存在！")
        column_index = self.header_index[column_name]
        return self.has_value_in_column_index(column_index, value)

    def has_value_in_column_index(self, column_index, value):
        """
        判断指定列序号中是否存在某个值
        """
        if column_index < 0 or column_index >= len(self.headers):
            raise IndexError(f"列索引 {column_index} 超出范围！")
        for row in self.data:
            if row[column_index] == value:
                return True
        return False

    def __repr__(self):
        return f"<CSVProcessor: {len(self.data)} rows, {len(self.header_index)} columns>"
