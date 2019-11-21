all: compare

compare: media/seq.gif media/omp.gif
	cmp media/*.gif
	
dist/%: %.c shared.h main.h
	gcc -march=ivybridge -ffast-math -O3 -Ofast -o $@ $< -lm -fopenmp 

media/%.gif: dist/% makegif.py
	rm -rf results/*
	$<
	python3 makegif.py "$@" > /dev/null 2>&1 
	gifsicle -O3 --batch "$@" > /dev/null 2>&1

clean:
	rm -f dist/*
	touch dist/.gitignore
	rm -f results/*.csv
	rm -f media/*.gif
	rm -f *.gif