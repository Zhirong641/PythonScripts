import csv
import random

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

    def get_rows_count(self):
        """
        获取所有行的数量
        """
        return len(self.data)

    def get_cols_count(self):
        """
        获取所有列的数量
        """
        return len(self.headers)

    def get_row(self, row_index):
        """
        获取指定行的所有数据
        """
        if row_index < 0 or row_index >= len(self.data):
            raise IndexError(f"行索引 {row_index} 超出范围！")
        return self.data[row_index]
    
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
    
    def get_rows_by_value(self, column_name, value, exact=True):
        """
        获取指定列中包含指定值的所有行
        column_name: 列名
        value: 要匹配的值
        exact: True 表示完全匹配，False 表示部分匹配（包含关系）
        返回：所有匹配的行（list of list）
        """
        if column_name not in self.header_index:
            raise KeyError(f"列名 '{column_name}' 不存在！")
        column_index = self.header_index[column_name]
        return self.get_rows_by_value_index(column_index, value, exact)

    def get_rows_by_value_index(self, column_index, value, exact=True):
        """
        获取指定列序号中包含指定值的所有行
        column_index: 列序号
        value: 要匹配的值
        exact: True 表示完全匹配，False 表示部分匹配（包含关系）
        返回：所有匹配的行（list of list）
        """
        if column_index < 0 or column_index >= len(self.headers):
            raise IndexError(f"列索引 {column_index} 超出范围！")
        
        if exact:
            return [row for row in self.data if row[column_index] == value]
        else:
            return [row for row in self.data if value in row[column_index]]

    def shuffle_rows(self):
        """
        随机打乱所有行的顺序
        """
        random.shuffle(self.data)

    def get_rows_by_range(self, start, length):
        """
        获取从 start 行开始的指定行数的所有行
        start: 起始行索引（从 0 开始）
        length: 行数
        返回：行的列表
        """
        if start < 0 or start + length > len(self.data):
            raise IndexError(f"索引范围 [{start}, {start + length - 1}] 超出行数范围 0-{len(self.data)-1}")
        if length <= 0:
            raise ValueError(f"行数 {length} 必须大于 0")

        return self.data[start:start + length]  # Python 切片是左闭右开，所以 end+1

    def __repr__(self):
        return f"<CSVProcessor: {len(self.data)} rows, {len(self.header_index)} columns>"
