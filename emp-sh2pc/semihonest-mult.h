#ifndef EMP_SEMIHONEST_MULT_H__
#define EMP_SEMIHONEST_MULT_H__
#include "emp-sh2pc/sh_gen.h"
#include "emp-sh2pc/sh_eva.h"

namespace emp {

template<typename IO>
class SHGen{
public:
  IO *io;
  HalfGateGen<IO> *t;
  CircuitExecution *lcirc_exec;
  ProtocolExecution *lprot_exec;
  block delta_used;

  SHGen(IO *ioObj, int batch_size = 1024*16) {
    io = ioObj;
    t = new HalfGateGen<IO>(io);
    lcirc_exec = t;
    lprot_exec = new SemiHonestGen<IO>(io, t);
    delta_used = t->delta;
  }

  void setup_execution_env(){
    CircuitExecution::circ_exec = lcirc_exec;
		ProtocolExecution::prot_exec = lprot_exec;
  }

  ~SHGen() {
    delete lcirc_exec;
  	delete lprot_exec;
  }

};

template<typename IO>
class SHEval{
public:
  NetIO *io;
  HalfGateEva<IO> *t;
  CircuitExecution *lcirc_exec;
  ProtocolExecution *lprot_exec;

  SHEval(IO *ioObj, int batch_size = 1024*16) {
    io = ioObj;
    t = new HalfGateEva<IO>(io);
		lcirc_exec = t;
		lprot_exec = new SemiHonestEva<IO>(io, t);
  }

  void setup_execution_env(){
    CircuitExecution::circ_exec = lcirc_exec;
		ProtocolExecution::prot_exec = lprot_exec;
  }

  ~SHEval() {
    delete lcirc_exec;
    delete lprot_exec;
  }
};
}
#endif
