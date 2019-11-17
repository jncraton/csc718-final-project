all: animation.gif

%: %.c
	gcc -O2 -o $@ $<

results/0.csv: seq
	./seq

animation.gif: results/0.csv
	python3 makegif.py

clean:
	rm -f seq
	rm -f results/*