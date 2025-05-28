"""
Usage:
cd results; python csv2tex.py

Input:
tables/exp*.csv

Output:
tex/exp*.tex
"""
from os.path import isfile, join

CSV_FILES = [
    'exp12.csv',
    'exp15.csv',
    'exp32.csv',
    'exp35.csv',
]

def main():
    for csv_file in CSV_FILES:
        csv_path = join('tables', csv_file)
        tex_path = join('tex', csv_file.replace('.csv', '.tex'))
        print(csv_file, isfile(csv_path))

        with open(csv_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]  # get rid of header
            lines = [line.replace('\n', ' \\\\\n').replace(',', ' & ').replace('True', '\\True').replace('False', '\\False') for line in lines]
        print(lines)

        with open(tex_path, 'w') as f:
            for line in lines:
                f.write(line)
        print(f'wrote file to {tex_path}')


if __name__ == '__main__':
    main()