CC:=clang
CFLAGS:= -Wall -Wextra -std=gnu11
DFLAGS:= -g -Og -fno-omit-frame-pointer
RFLAGS:= -Ofast -DNDEBUG 
SRCS:= nn.c mat.c

.PHONY: release debug unzip zip clean

release:
	$(CC) $(CFLAGS) $(RFLAGS) $(SRCS) cgan.c -lm -o cgan

debug:
	$(CC) $(CFLAGS) $(DFLAGS) $(SRCS) cgan.c -lm -o cgan

test: 
	$(CC) $(CFLAGS) $(DFLAGS) $(SRCS) test.c -lm -o test
	./test
	-rm test

unzip: 
	unzip data.zip
	rm data.zip

zip:
	zip data.zip mnist*.csv
	rm mnist* 


clean:
	-rm cgan



	
