all: media/seq.gif media/omp.gif

dist/%: %.c
	gcc -ffast-math -O3 -o $@ $< -lm -fopenmp 

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