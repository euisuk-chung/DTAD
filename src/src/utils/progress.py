import sys

def printProgressBar(iteration, total, prefix = 'Progress', suffix = 'Complete',\
                      decimals = 1, length = 70, fill = '█'):
    # 작업의 진행상황을 표시
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    _string_out = '\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix)
    sys.stdout.write(_string_out)
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

if __name__ == '__main__':

    for i in range(100):
        printProgressBar(i+1, 100)