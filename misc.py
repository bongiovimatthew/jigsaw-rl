from multiprocessing import Pool

def f(x):
    return x*x
print("dfasdfdf")
if __name__ == '__main__':
    p = Pool(2)
    print(p.map(f, [1, 2, 3]))