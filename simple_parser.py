# Utility 

# Crea un dictionary da un pbtxt file
def dict_from_pbtxt_file(fname):
    """Given label map proto returns categories list compatible with eval.

    Questa funzione ritorna un dizionario

    {1: 'label1', 2:'label2', ... }

    estraendoli da un file pbtxt 
  
    Args:
        fname: percorso del file pbtxt
      
    Returns:
        label_map: dictionaries representing all possible categories.
    """
    lines = [line.rstrip('\n').strip() for line in open(fname)]
    label_map = {}
    curr_label = ''
    curr_id = 0

    for l in lines:
    
        if l.startswith( 'display_name: '):
            curr_label = l.split(' ')[1]

        if l.startswith( 'id: '):
            curr_id = int(l.split(' ')[1])

        if l.startswith( '}'):
            # print(curr_id, curr_label)
            label_map[curr_id] = curr_label.replace("\"", "")

    return label_map

"""
args_prototxt = 'c:/nn/mscoco_label_map.pbtxt'

lbm = dict_from_pbtxt_file(args_prototxt)
print('-------------------------------------------------------------')
print(lbm)
print('-------------------------------------------------------------')


lines = [line.rstrip('\n').strip() for line in open(args_prototxt)]
#print(lines)

label_map = {}
curr_label = ''
curr_id = 0

for l in lines:
    #if l.startswith( 'item' ):
        #print('n')

    if l.startswith( 'display_name: '):
        #print('display_name')
        #print(l.split(' '))
        curr_label = l.split(' ')[1]

    #if l.startswith( 'name: '):
        #print('name')
        #print(l.split(' ')[1])

    if l.startswith( 'id: '):
        #print('id')
        #print(l.split(' ')[1])   
        curr_id = int(l.split(' ')[1])

    if l.startswith( '}'):
        #print('end')
        print(curr_id, curr_label)
        label_map[curr_id] = curr_label.replace("\"", "")
        #label_map.insert( curr_id, curr_label)


print('array ----------')
print(label_map)

print(label_map[38])
"""