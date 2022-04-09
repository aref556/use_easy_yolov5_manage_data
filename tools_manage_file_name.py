import os

def rename_file_from_directory(path_directory: str, first_charector_name: str):
    list_file = os.listdir(path_directory)
    for i, namefile in enumerate(list_file):
        # print('id {} : {}'.format(i, namefile))
        id = i+1
        
        os.rename('{}/{}'.format(path_directory, namefile), '{}/{}-{}.jpg'.format(path_directory, first_charector_name, id))
        
        print('old_name : {} / new_name: {}-{}.jpg'.format(namefile, first_charector_name, id))
    

def main():
    cur_dir = os.getcwd()
    PATH_DIRECTORY = '{}/cars/c9'.format(cur_dir)
    FIRST_CHERECTER = 'c9'
    rename_file_from_directory(PATH_DIRECTORY, FIRST_CHERECTER)
    
if __name__ == '__main__':
    print('cur_dir : {}'.format(os.getcwd()))
    main()
        
        
    