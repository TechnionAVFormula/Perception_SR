import cProfile
import pstats
import io
from test import main
pr = cProfile.Profile()
pr.enable()

main()

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test_performance.txt', 'w+') as f:
    f.write(s.getvalue())