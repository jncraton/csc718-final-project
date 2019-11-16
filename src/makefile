all: seq

%: %.c
	gcc -O2 -o $@ $<

results/0.csv: seq
	./seq

clean:
	rm -f seq
	rm -f results/*