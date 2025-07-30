from CSVProcessor import CSVProcessor

def test_csv_processor():
    # 创建 CSVProcessor 实例
    processor = CSVProcessor('/mnt/shared/data/cglist.csv', has_header=False)
    
    # 打印数据和表头
    print(processor)
    
    # 测试列索引查找
    try:
        print("Has value in column index 0:", processor.has_value_in_column_index(4, '727768')) # 假设第5列有值 '727768'
        print("Has value in column index 0:", processor.has_value_in_column_index(4, '123456')) # 假设第5列没有值 '123456'

        print("Has value in column 'col4':", processor.has_value_in_column('col4', '727768'))
        print("Has value in column 'col0':", processor.has_value_in_column('col4', '123456'))

        print("Value at row 0, column 'col4':", processor.get_value(0, 'col4'))  # 假设第5列有值
        print("Value at row 0, column index 4", processor.get_value_by_index(0, 4))  # 假设第5列有值
    except IndexError as e:
        print("IndexError:", e)


test_csv_processor()