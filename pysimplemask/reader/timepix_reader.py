import numpy as np
import struct as struct
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time
import os


def get_saxs(fname="./208K_7200s_s50ms_cooling.raw", max_shutter=1e8, 
             offset=0, end=None):

    shutter_count = 0
    total_event_count = 0

    img = np.zeros((512, 512), dtype=np.int32)  # image size

    # T_Array = np.zeros(0, dtype=np.uint64)
    timepix_dtype = [('time_local', 'i8'), ('X', 'i2'), ('Y', 'i2')]

    infile = open(fname, 'rb')
    infile.seek(offset)

    location = offset
    # infinite cycle here till reach the end of the file

    while True:
        buffer = infile.read(64)  # Read 64 bytes
        if len(buffer) < 64:
            if len(buffer) != 0:
                print('ERROR: Should not be reading that new buffer here')
            break

        if buffer[0] != 0x4A or buffer[63] != 0x4B:
            print('ERROR: Need to synchronize to next header start')
            break
        # unpack_from not sure ask

        global_time, shutter_number, time_from_trig, event_count, ClkDvdr = \
            struct.unpack_from('QIIIH', buffer[1:23])

        shutter_count += 1
        total_event_count += event_count

        buffer = infile.read(event_count * 12)

        location += event_count * 12
        if end is not None and location > end:
            break

        # create buffer to read times of the photons
        # T_tmp = np.zeros(Ncnts, dtype=np.uint64)

        res = np.frombuffer(buffer, timepix_dtype, count=event_count)
        # lTmp, X, Y = res['time_local'], res['X'], res['Y']
        X, Y = res['X'], res['Y']

        # Y = Y.astype(np.uint32)
        # xy = (Y << 9) + X
        # tot = np.bincount(xy, minlength=512*512+1)[1:]
        # img += tot.reshape(512, 512)

        # the maximal number of photons on a pixel during a shutter period is
        # 1. use the index directly 
        img[(Y, X)] += 1

        if shutter_count >= max_shutter:
            break

    infile.close()
    # print(max_count)

    return img


def split_file(fname, num=8):
    size = os.path.getsize(fname)
    start = np.linspace(0, size, num, endpoint=False, dtype=np.int64)
    start = (start // 4) * 4

    infile = open(fname, 'rb')
    
    def is_heander(offset):
        infile.seek(offset)
        buffer = infile.read(64)  # Read 64 bytes

        if len(buffer) < 64:
            raise ValueError

        if buffer[0] == 0x4A and buffer[63] == 0x4B:
            return True
        else:
            return False
    
    for n in range(num-1):
        guess = start[n + 1]
        while not is_heander(guess):
            guess += 4
        start[n + 1] = guess

    end = np.zeros_like(start)    
    end[:-1] = start[1:]
    end[-1] = size
    return start, end  


def get_saxs_wrap(kwargs):
    return get_saxs(**kwargs)


def get_saxs_mp(fname, num_thread=8, **kwargs):
    start, end = split_file(fname, num_thread)
    p = Pool(num_thread)
    args = [{'fname': fname, 'offset': start[n], 'end': end[n]} 
            for n in range(len(start))]
    result = p.map(get_saxs_wrap, args)
    result = np.array(result).sum(axis=0)
    return result


if __name__ == '__main__':
    t0 = time.perf_counter()
    # img = get_saxs()
    img = get_saxs_mp('./208K_7200s_s50ms_cooling.raw')
    print(time.perf_counter() - t0)
    img = np.log10(img + 1)
    img[img == 0] = np.nan
    plt.imshow(img, cmap=plt.cm.jet)
    plt.show()
