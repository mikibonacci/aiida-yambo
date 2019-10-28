from __future__ import print_function
StructureData = DataFactory('structure')
cell = [[31.7506200000, 0.0000000000, 0.0000000000],
        [0.0000000000, 21.1670800000, 0.0000000000],
        [0.0000000000, 0.0000000000, 4.2689977570]]
s = StructureData(cell=cell)
s.append_atom(
    position=[-4.6124367660, 0.0000000000, -1.1971046460], symbols="C")
s.append_atom(
    position=[-4.6124367660, 0.0000000000, 1.2171046460], symbols="C")
s.append_atom(
    position=[-3.6523182400, 0.0000000000, -0.6721985250], symbols="H")
s.append_atom(
    position=[-3.6523182400, 0.0000000000, 0.6921985250], symbols="H")
s.append_atom(
    position=[-2.4520668990, 0.0000000000, -1.4092894790], symbols="H")
s.append_atom(
    position=[-2.4520668990, 0.0000000000, 1.4292894790], symbols="H")
s.append_atom(
    position=[-1.2292046550, 0.0000000000, -0.7024027910], symbols="H")
s.append_atom(
    position=[-1.2292046550, 0.0000000000, 0.7224027910], symbols="H")
s.append_atom(
    position=[0.0000000000, 0.0000000000, -1.4182198960], symbols="H")
s.append_atom(position=[0.0000000000, 0.0000000000, 1.4382198960], symbols="H")
s.append_atom(
    position=[1.2292046550, 0.0000000000, -0.7024027910], symbols="H")
s.append_atom(position=[1.2292046550, 0.0000000000, 0.7224027910], symbols="H")
s.append_atom(
    position=[2.4520668990, 0.0000000000, -1.4092894790], symbols="H")
s.append_atom(position=[2.4520668990, 0.0000000000, 1.4292894790], symbols="H")
s.append_atom(
    position=[3.6523182400, 0.0000000000, -0.6721985250], symbols="H")
s.append_atom(position=[3.6523182400, 0.0000000000, 0.6921985250], symbols="H")
s.append_atom(
    position=[4.6124367660, 0.0000000000, -1.1971046460], symbols="C")
s.append_atom(position=[4.6124367660, 0.0000000000, 1.2171046460], symbols="C")

print(s.store())
