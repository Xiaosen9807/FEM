import sys

print ('The command line arguments are:')
for i in sys.argv:
    print ('xxx', i)

print ('\n\nThe PYTHONPATH is', sys.path[1], '\n')

filename = sys.path[0]+'/stdFEM/input/'+'bar.json'
print('/Users/qinxusen/Desktop/FEM/stdFEM/input/bar.json')
print(filename)

def test():
    dun = diff(un, x)
    Bn = integrate((dun**2+un**2), (x, 0, 1))
    An = (1/2*Bn)**0.5
