# converte le annotazioni xml in una tabella csv per
# la successiva conversione in TFRecord


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print('reading:', xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            print('add', value)
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Taking command line arguments from users
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_xml', help='define the input xml file', type=str, required=True)
    parser.add_argument('-out', '--output_csv', help='define the output file ', type=str, required=True)
    args = parser.parse_args()
    '''

   
    args_input_xml = 'C:/nn/dataset/source'
    args_output_csv = 'C:/nn/dataset/csv/1C-blu-900x900.csv'

    xml_df = xml_to_csv(args_input_xml)
    xml_df.to_csv(args_output_csv, index=None)
    print('Successfully converted xml to csv.')


main()