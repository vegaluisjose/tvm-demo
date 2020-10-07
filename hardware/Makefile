default: compile_cc

compile_verilog:
	verilator --cc accelerator.v wrapper.v --prefix Top --Mdir out \
	-Wno-BLKANDNBLK \
	-Wno-PINMISSING \
	-Wno-STMTDLY \
	-Wno-WIDTH \
	-Wno-UNOPTFLAT


compile_cc: compile_verilog
	g++ -O2 -std=c++14 \
	-I out \
	-I /usr/local/share/verilator/include \
	/usr/local/share/verilator/include/verilated.cpp \
	out/Top__Slow.cpp \
	out/Top__Syms.cpp \
	out/Top.cpp \
	device.cc \
	test.cc