CC:=cc
CFLAGS:= -Wall -Wextra -lm -lgsl -lgslcblas 
DFLAGS:= $(CFLAGS) -g -O0 -fno-omit-frame-pointer -fsanitize=undefined,address
RFLAGS:= $(CFLAGS) -Ofast -DNDEBUG 
SRCS:= nn.c 


release:
	$(CC) nn.c cgan.c -o cgan $(RFLAGS)

debug:
	$(CC) nn.c cgan.c -o cgan $(DFLAGS)

test: 
	$(CC) nn.c test.c -o test $(DFLAGS)
	./test
	-rm test

unzip: 
	unzip data.zip
	rm data.zip

zip:
	zip data.zip train-*
	rm train-*


clean:
	-rm cgan



	
