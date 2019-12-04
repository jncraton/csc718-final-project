all: compare

compare: media/seq.gif media/omp.gif media/cuda.gif
	cmp $< media/omp.gif
	cmp $< media/cuda.gif

dist/cuda: cuda.cu
	nvcc --compiler-options -ffast-math --compiler-options -O3 --compiler-options -Ofast $< -o $@ --compiler-options -lm --compiler-options -fopenmp

dist/%: %.c shared.h main.h
	g++ -Wall -Wextra -march=ivybridge -ffast-math -O3 -Ofast -o $@ $< -lm -fopenmp

media/%.gif: dist/% render.py
	rm -rf results/*
	$<
	python3 render.py "$@" > /dev/null 2>&1 
	gifsicle -O3 --batch "$@" > /dev/null 2>&1

clean:
	rm -f dist/*
	touch dist/.gitignore
	rm -f results/*.csv
	rm -f media/*.gif
	rm -f *.gif