OfficeHome = {'53': 'Alarm_Clock', '39': 'Backpack', '26': 'Batteries', '47': 'Bed', '28': 'Bike', '0': 'Bottle', '3': 'Bucket',
              '2': 'Calculator', '41': 'Calendar', '35': 'Candles', '49': 'Chair', '43': 'Clipboards', '21': 'Computer', '52': 'Couch',
              '16': 'Curtains', '18': 'Desk_Lamp', '14': 'Drill', '6': 'Eraser', '11': 'Exit_Sign', '45': 'Fan', '59': 'File_Cabinet',
              '15': 'Flipflops', '61': 'Flowers', '1': 'Folder', '48': 'Fork', '17': 'Glasses', '9': 'Hammer', '4': 'Helmet', '27': 'Kettle',
              '29': 'Keyboard', '32': 'Knives', '60': 'Lamp_Shade', '5': 'Laptop', '64': 'Marker', '40': 'Monitor', '42': 'Mop', '33': 'Mouse',
              '44': 'Mug', '63': 'Notebook', '57': 'Oven', '51': 'Pan', '34': 'Paper_Clip', '56': 'Pen', '36': 'Pencil', '50': 'Postit_Notes',
              '58': 'Printer', '22': 'Push_Pin', '20': 'Radio', '37': 'Refrigerator', '7': 'Ruler', '55': 'Scissors', '13': 'Screwdriver',
              '54': 'Shelf', '8': 'Sink', '62': 'Sneakers', '38': 'Soda', '30': 'Speaker', '19': 'Spoon', '24': 'TV', '31': 'Table',
              '12': 'Telephone', '46': 'ToothBrush', '23': 'Toys', '25': 'Trash_Can', '10': 'Webcam'}
PACS = {'0': 'dog', '1': 'elephant', '2': 'giraffe', '3': 'guitar', '4': 'horse', '5': 'house', '6': 'person'}
VLCS = {'0': 'bird', '1': 'car', '2': 'chair', '3': 'dog', '4': 'person'}
digits_dg = {str(i):str(i) for i in range(10)}
DomainNet = {'0': 'aircraft_carrier', '1': 'airplane', '2': 'alarm_clock', '3': 'ambulance', '4': 'angel', '5': 'animal_migration', '6': 'ant',
             '7': 'anvil', '8': 'apple', '9': 'arm', '10': 'asparagus', '11': 'axe', '12': 'backpack', '13': 'banana', '14': 'bandage',
             '15': 'barn', '16': 'baseball', '17': 'baseball_bat', '18': 'basket', '19': 'basketball', '20': 'bat', '21': 'bathtub',
             '22': 'beach', '23': 'bear', '24': 'beard', '25': 'bed', '26': 'bee', '27': 'belt', '28': 'bench', '29': 'bicycle',
             '30': 'binoculars', '31': 'bird', '32': 'birthday_cake', '33': 'blackberry', '34': 'blueberry', '35': 'book', '36': 'boomerang',
             '37': 'bottlecap', '38': 'bowtie', '39': 'bracelet', '40': 'brain', '41': 'bread', '42': 'bridge', '43': 'broccoli', '44': 'broom',
             '45': 'bucket', '46': 'bulldozer', '47': 'bus', '48': 'bush', '49': 'butterfly', '50': 'cactus', '51': 'cake', '52': 'calculator',
             '53': 'calendar', '54': 'camel', '55': 'camera', '56': 'camouflage', '57': 'campfire', '58': 'candle', '59': 'cannon',
             '60': 'canoe', '61': 'car', '62': 'carrot', '63': 'castle', '64': 'cat', '65': 'ceiling_fan', '66': 'cello', '67': 'cell_phone',
             '68': 'chair', '69': 'chandelier', '70': 'church', '71': 'circle', '72': 'clarinet', '73': 'clock', '74': 'cloud',
             '75': 'coffee_cup', '76': 'compass', '77': 'computer', '78': 'cookie', '79': 'cooler', '80': 'couch', '81': 'cow', '82': 'crab',
             '83': 'crayon', '84': 'crocodile', '85': 'crown', '86': 'cruise_ship', '87': 'cup', '88': 'diamond', '89': 'dishwasher',
             '90': 'diving_board', '91': 'dog', '92': 'dolphin', '93': 'donut', '94': 'door', '95': 'dragon', '96': 'dresser', '97': 'drill',
             '98': 'drums', '99': 'duck', '100': 'dumbbell', '101': 'ear', '102': 'elbow', '103': 'elephant', '104': 'envelope', '105': 'eraser',
             '106': 'eye', '107': 'eyeglasses', '108': 'face', '109': 'fan', '110': 'feather', '111': 'fence', '112': 'finger',
             '113': 'fire_hydrant', '114': 'fireplace', '115': 'firetruck', '116': 'fish', '117': 'flamingo', '118': 'flashlight',
             '119': 'flip_flops', '120': 'floor_lamp', '121': 'flower', '122': 'flying_saucer', '123': 'foot', '124': 'fork', '125': 'frog',
             '126': 'frying_pan', '127': 'garden', '128': 'garden_hose', '129': 'giraffe', '130': 'goatee', '131': 'golf_club', '132': 'grapes',
             '133': 'grass', '134': 'guitar', '135': 'hamburger', '136': 'hammer', '137': 'hand', '138': 'harp', '139': 'hat',
             '140': 'headphones', '141': 'hedgehog', '142': 'helicopter', '143': 'helmet', '144': 'hexagon', '145': 'hockey_puck',
             '146': 'hockey_stick', '147': 'horse', '148': 'hospital', '149': 'hot_air_balloon', '150': 'hot_dog', '151': 'hot_tub',
             '152': 'hourglass', '153': 'house', '154': 'house_plant', '155': 'hurricane', '156': 'ice_cream', '157': 'jacket', '158': 'jail',
             '159': 'kangaroo', '160': 'key', '161': 'keyboard', '162': 'knee', '163': 'knife', '164': 'ladder', '165': 'lantern',
             '166': 'laptop', '167': 'leaf', '168': 'leg', '169': 'light_bulb', '170': 'lighter', '171': 'lighthouse', '172': 'lightning',
             '173': 'line', '174': 'lion', '175': 'lipstick', '176': 'lobster', '177': 'lollipop', '178': 'mailbox', '179': 'map',
             '180': 'marker', '181': 'matches', '182': 'megaphone', '183': 'mermaid', '184': 'microphone', '185': 'microwave', '186': 'monkey',
             '187': 'moon', '188': 'mosquito', '189': 'motorbike', '190': 'mountain', '191': 'mouse', '192': 'moustache', '193': 'mouth',
             '194': 'mug', '195': 'mushroom', '196': 'nail', '197': 'necklace', '198': 'nose', '199': 'ocean', '200': 'octagon',
             '201': 'octopus', '202': 'onion', '203': 'oven', '204': 'owl', '205': 'paintbrush', '206': 'paint_can', '207': 'palm_tree',
             '208': 'panda', '209': 'pants', '210': 'paper_clip', '211': 'parachute', '212': 'parrot', '213': 'passport', '214': 'peanut',
             '215': 'pear', '216': 'peas', '217': 'pencil', '218': 'penguin', '219': 'piano', '220': 'pickup_truck', '221': 'picture_frame',
             '222': 'pig', '223': 'pillow', '224': 'pineapple', '225': 'pizza', '226': 'pliers', '227': 'police_car', '228': 'pond',
             '229': 'pool', '230': 'popsicle', '231': 'postcard', '232': 'potato', '233': 'power_outlet', '234': 'purse', '235': 'rabbit',
             '236': 'raccoon', '237': 'radio', '238': 'rain', '239': 'rainbow', '240': 'rake', '241': 'remote_control', '242': 'rhinoceros',
             '243': 'rifle', '244': 'river', '245': 'roller_coaster', '246': 'rollerskates', '247': 'sailboat', '248': 'sandwich', '249': 'saw',
             '250': 'saxophone', '251': 'school_bus', '252': 'scissors', '253': 'scorpion', '254': 'screwdriver', '255': 'sea_turtle',
             '256': 'see_saw', '257': 'shark', '258': 'sheep', '259': 'shoe', '260': 'shorts', '261': 'shovel', '262': 'sink',
             '263': 'skateboard', '264': 'skull', '265': 'skyscraper', '266': 'sleeping_bag', '267': 'smiley_face', '268': 'snail',
             '269': 'snake', '270': 'snorkel', '271': 'snowflake', '272': 'snowman', '273': 'soccer_ball', '274': 'sock', '275': 'speedboat',
             '276': 'spider', '277': 'spoon', '278': 'spreadsheet', '279': 'square', '280': 'squiggle', '281': 'squirrel', '282': 'stairs',
             '283': 'star', '284': 'steak', '285': 'stereo', '286': 'stethoscope', '287': 'stitches', '288': 'stop_sign', '289': 'stove',
             '290': 'strawberry', '291': 'streetlight', '292': 'string_bean', '293': 'submarine', '294': 'suitcase', '295': 'sun', '296': 'swan',
             '297': 'sweater', '298': 'swing_set', '299': 'sword', '300': 'syringe', '301': 'table', '302': 'teapot', '303': 'teddy-bear',
             '304': 'telephone', '305': 'television', '306': 'tennis_racquet', '307': 'tent', '308': 'The_Eiffel_Tower',
             '309': 'The_Great_Wall_of_China', '310': 'The_Mona_Lisa', '311': 'tiger', '312': 'toaster', '313': 'toe', '314': 'toilet',
             '315': 'tooth', '316': 'toothbrush', '317': 'toothpaste', '318': 'tornado', '319': 'tractor', '320': 'traffic_light',
             '321': 'train', '322': 'tree', '323': 'triangle', '324': 'trombone', '325': 'truck', '326': 'trumpet', '327': 't-shirt',
             '328': 'umbrella', '329': 'underwear', '330': 'van', '331': 'vase', '332': 'violin', '333': 'washing_machine', '334': 'watermelon',
             '335': 'waterslide', '336': 'whale', '337': 'wheel', '338': 'windmill', '339': 'wine_bottle', '340': 'wine_glass',
             '341': 'wristwatch', '342': 'yoga', '343': 'zebra', '344': 'zigzag'}

OfficeHome31 = {'5': 'calculator', '24': 'ring_binder', '21': 'printer', '11': 'keyboard', '26': 'scissors', '12': 'laptop_computer',
                '16': 'mouse', '15': 'monitor', '17': 'mug', '29': 'tape_dispenser', '19': 'pen', '1': 'bike', '23': 'punchers',
                '0': 'back_pack', '8': 'desktop_computer', '27': 'speaker', '14': 'mobile_phone', '18': 'paper_notebook', '25': 'ruler',
                '13': 'letter_tray', '9': 'file_cabinet', '20': 'phone', '3': 'bookcase', '22': 'projector', '28': 'stapler', '30': 'trash_can',
                '2': 'bike_helmet', '10': 'headphones', '7': 'desk_lamp', '6': 'desk_chair', '4': 'bottle'}
Visda17 = {'0': 'aeroplane', '1': 'bicycle', '2': 'bus', '3': 'car', '4': 'horse', '5': 'knife', '6': 'motorcycle', '7': 'person', '8': 'plant',
           '9': 'skateboard', '10': 'train', '11': 'truck'}


if __name__ == '__main__':
    dataset = digits_dg
    idx = [int(k) for k in dataset.keys()]
    for i in idx:
        print(dataset[str(i)])
