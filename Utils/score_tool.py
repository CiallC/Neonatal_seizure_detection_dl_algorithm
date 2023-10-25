import numpy as np
import sys
import pdb
from scipy import fix, signal

avfilllength = 61  # AD this variable is not used
# collar = 18 # for 8 seconds collar = 30 for 16 seconds = 26, for 32 seconds
resp = 145 * 4  # AD this variable is used

threshold_pass = 0
threshold_reject = 1


def simpoly(x, y):
    """
    A function that calculates the area of a 2-D simple polygon (no matter concave or convex)
    Must name the vertices in sequence (i.e., clockwise or counterclockwise)
    Inputs must be float type
    Formula used: http://en.wikipedia.org/wiki/Polygon#Area_and_centroid
    Definition of "simply polygon": http://en.wikipedia.org/wiki/Simple_polygon
    Input x: x-axis coordinates of vertex array
          y: y-axis coordinates of vertex array
    Output: polygon area
    """

    ind_arr = np.arange(len(x)) - 1  # for indexing convenience
    s = 0
    for ii in ind_arr:
        s = s + (x[ii] * y[ii + 1] - x[ii + 1] * y[ii])

    return abs(s) * 0.5


if __name__ == "__main__":
    usage_str = 'Usage:\npython polyarea.py c 1 2 4 8 3 5 \npython polyarea.py f coord.txt\npython polyarea.py help'

    if len(sys.argv) == 1:
        print("Error: argument input needed")
        print(usage_str)
        exit()

    if sys.argv[1].lower() == 'c':
        # Recognize the inputs as the coordinates
        if len(sys.argv) <= 7:
            print("Error: at least three 2-D points needed for a valid polygon")
            print("Exiting...")
            exit()
        elif np.mod(len(sys.argv[2:]), 2) != 0:
            print("Error: the number of input arguments should be even")
            print("Exiting...")
            exit()
        else:
            print("This polygon has", (len(sys.argv) - 2) / 2, "vertices")

        a = np.zeros(len(sys.argv) - 2)  # the default a.dtype.name is float64

        ind = 0
        for coord in sys.argv[2:]:
            a[ind] = float(eval(coord))  # convert the input arguments to "float" type
            ind = ind + 1

        b = a.reshape(-1, 2).copy()
        x = b[:, 0].copy()  # get x coords
        y = b[:, 1].copy()  # get y coords

        print("The area of this polygon is", simpoly(x, y))

    elif sys.argv[1].lower() == 'f':
        # Get the input from a file
        # in which the 1st column is x coordinates, and the 2nd column is y coordinates
        d = np.loadtxt(sys.argv[2])
        x = d[:, 0]
        y = d[:, 1]
        print("The area of this polygon is", simpoly(x, y))

    elif sys.argv[1].lower() == 'help':
        print(usage_str)
        exit()

    else:
        print("Error need input arg either c (command line) or f (file)")
        print("Exiting")
        exit()


def enframe(x, win, inc):
    nx = len(x)
    try:
        nwin = len(win)
    except TypeError:
        nwin = 1
    if nwin == 1:
        length = win
    else:
        length = nwin

    nf = int(fix((nx - length + inc) // inc))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) +
           np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    # f = signal.detrend(f, type='constant')
    # no_win, _ = f
    return f


def calc_roc(D_val, epoch_map, epoch_length):
    collar = int(30 - (epoch_length - 8) / 2)

    D_val = np.asarray(D_val)
    ctr = 0
    sens = []
    spec = []
    prec = []
    prec2 = []

    float_values = [float(x) / 10000 for x in range(0, 10025, 25)]

    for value in float_values:
        # pdb.set_trace()
        decision = np.zeros((1, len(D_val)))
        for idx in np.where(D_val >= 1 - value)[0]:
            decision[0, idx] = int(1)

        shiftmean = int((11 - 1) / 2)
        my_dec = np.zeros((int(len(decision[0]) + shiftmean * 2)))
        pt1 = decision[:, :shiftmean]
        my_dec[:len(pt1[0])] = np.fliplr([pt1])[0]
        my_dec[len(pt1[0]):len(pt1[0]) + int(len(decision[0]))] = decision[0]
        pt2 = decision[:, -shiftmean:]
        my_dec[len(pt1[0]) + int(len(decision[0])):] = np.fliplr([pt2])[0]

        decision = np.zeros((1, len(D_val)))
        for zxc in range(1):
            aaa = enframe(np.array(my_dec), 11, 1)
            for idx in np.where(np.sum(aaa, 1) > 5)[0]:
                decision[0, idx] = 1
        idxtmp = np.where(decision[0] == 1)[0]
        mymean = 0
        if value < 0.9:
            for kk in range(len(idxtmp)):
                if kk > 0 and idxtmp[kk] - idxtmp[kk - 1] == 1:
                    mymean = mymean
                else:
                    startx = idxtmp[kk] - resp
                    if startx <= 0:
                        startx = 0
                    rng = np.setdiff1d(range(startx, idxtmp[kk]), idxtmp)
                    if len(rng) < 1:
                        mymean = 0
                    else:
                        mymean = np.nanmean(D_val[rng])
                if mymean > 0:
                    if not D_val[idxtmp[kk]] + value - 1 * mymean > 1:
                        decision[0, idxtmp[kk]] = 0

        shiftmean = int((11 - 1) / 2)
        my_dec = np.zeros((int(len(decision[0])) + shiftmean * 2))
        pt1 = decision[:, :shiftmean]
        my_dec[:len(pt1[0])] = np.fliplr([pt1])[0]
        my_dec[len(pt1[0]):len(pt1[0]) + int(len(decision[0]))] = decision[0]
        pt2 = decision[:, -shiftmean:]
        my_dec[len(pt1[0]) + int(len(decision[0])):] = np.fliplr([pt2])[0]

        decision = np.zeros((1, int(len(D_val))))
        for zxc in range(1):
            aaa = enframe(np.array(my_dec), 11, 1)
            for idx in np.where(np.sum(aaa, 1) > 5)[0]:
                decision[0, idx] = 1

        globdecision_max = decision

        tmp = np.where(np.array(globdecision_max[0]) == 1)
        finidx = []

        for kkk in range(collar + 1):
            a = tmp[0] - kkk  # [val-kkk for val in tmp[0]]
            b = tmp[0] + kkk  # [val +kkk for val in tmp[0]]
            finidx.extend(a)
            finidx.extend(b)
            finidx = list(set(finidx))
        negative_values = np.where(np.array(finidx) < 0)
        for neg in negative_values[0]:
            # print(neg)
            finidx[neg] = 0
        positive_values = np.where(np.array(finidx) >= int(len(decision[0])))
        for pos in positive_values[0]:
            finidx[pos] = len(decision[0]) - 1
        for pred_idx in finidx:
            globdecision_max[0][pred_idx] = 1

        TN = 0
        FN = 0
        FP = 0
        TP = 0
        calc_vec = epoch_map + globdecision_max * 2
        TN = TN + float(len(np.where(np.array(calc_vec) == 0)[0]))
        FN = FN + float(len(np.where(np.array(calc_vec) == 1)[0]))
        FP = FP + float(len(np.where(np.array(calc_vec) == 2)[0]))
        TP = TP + float(len(np.where(np.array(calc_vec) == 3)[0]))
        del calc_vec

        try:
            sens.append(float(TP / (TP + FN)))
        except ZeroDivisionError:
            sens.append(np.nan)
        try:
            spec.append(float(TN / (TN + FP)))
        except ZeroDivisionError:
            spec.append(np.nan)
        try:
            prec.append(float(TP / (TP + FP)))
        except ZeroDivisionError:
            prec.append(np.nan)
        try:
            prec2.append(float(TN / (TN + FN)))
        except ZeroDivisionError:
            prec2.append(np.nan)
        #        del TN, TP, FN, FP
        ctr = ctr + 1

    x = [0, 1]
    x.extend(spec)
    x.extend([0])
    y = [0, 0]
    y.extend(sens)
    y.extend([1])
    roc_area = simpoly(x, y)
    # print("Test ROC area = %f \n" % (roc_area))

    Nfix = 0.90
    N90 = np.where(np.asarray(spec) > Nfix)[0][-1]
    if N90 == 0:
        M90 = 0
        K90 = 0
    else:
        M90 = np.max(sens[:N90])
        K90 = np.max(spec[:N90])

    x = [Nfix, K90]
    x.extend(spec[:N90])
    x.extend([Nfix])

    y = [0, 0]
    y.extend(sens[:N90])
    y.extend([M90])

    roc_area90 = simpoly(x, y)

    # print("Test ROC90 area = %f \n" % (10*roc_area90))

    return (roc_area, roc_area90)