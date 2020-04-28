from collections import namedtuple
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing
import numpy as np
import os


#--------------------------------------------------------------------------------
# Args
#--------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Panoptic testing script")
parser.add_argument(
    "--inputs",
    type=str,
    default="images",
    help="iSAID dataset"
)
parser.add_argument(
    "--outputs",
    type=str,
    default="iSAID_id",
    help="converted images output directory"
)
parser.add_argument(
    "--noempty",
    action='store_true',
    help="ignore the ground truth which has no label but background"
)
args = parser.parse_args()

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    'm_color'       , # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
    # Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ), # background
    Label(  'background'           ,  0 ,       15 , 'void'            , 0       , False        , False        , (  0,  0,  0) , 0      ),
    Label(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    Label(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    Label(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63,  0) , 16128  ),
    Label(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    Label(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    Label(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    Label(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    Label(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    Label(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    Label(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    Label(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    Label(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    Label(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    Label(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    Label(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),
]

# instance classes id
i_class = list(range(1, 16))

# paths
inputs = args.inputs
outputs = args.outputs

# color to id
color2id = {}
for l in labels:
    color2id[l.color] = l.id

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def worker(name):
    ## open & cvt to array
    sp, ip = name + '_instance_color_RGB.png', name + '_instance_id_RGB.png'
    sem, ins = Image.open(sp), Image.open(ip)
    sem, ins = np.asarray(sem), np.asarray(ins)

    ## ignore empty ground truth
    if args.noempty and not np.any(sem) and not np.any(ins):
        print('ignore' + sp)
        return

    ## semantic image to class_id map
    # 3d array -> 2d tuple array
    sem_tuplemap = sem.view([(f'f{i}', sem.dtype) for i in range(sem.shape[-1])])[..., 0].astype('O')
    # map RGB to class id
    sem_idmap = np.vectorize(color2id.get)(sem_tuplemap)

    ## instance image to instance_id map
    # 3d array -> 2d tuple array
    ins_tuplemap = ins.view([(f'f{i}', ins.dtype) for i in range(ins.shape[-1])])[..., 0].astype('O')
    # map rgb to hex
    ins_idmap = np.vectorize(rgb_to_hex)(ins_tuplemap)
    # assign every unique hex to instance id
    ins_ids = list(np.unique(ins_idmap))
    ins_ids.remove('000000')
    ins_ids_dict = {key: idx for idx, key in enumerate(ins_ids, start=1)}
    ins_ids_dict['000000'] = 0
    # map hex to instance id
    ins_idmap = np.vectorize(ins_ids_dict.get)(ins_idmap)

    ## generate the output
    # create a canvas
    ins_new = np.zeros(ins.shape[:2], dtype=np.int32)
    # merge
    ins_new += sem_idmap * 1000 + ins_idmap

    # save
    Image.fromarray(ins_new).save(ip.replace(inputs, outputs))


def main():
    # make output dir if not exist
    try:
        os.mkdir(outputs)
    except:
        pass

    # files
    img_names = []
    for i in os.listdir(inputs):
        if 'instance' not in i:
            img_names.append(os.path.join(inputs, i.split('.')[0]))
    
    # do with progress bar
    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap(worker, img_names), total=len(img_names)))


if __name__=="__main__":
    main()
    print('Done!')
