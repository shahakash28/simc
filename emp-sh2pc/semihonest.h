#ifndef EMP_SEMIHONEST_H__
#define EMP_SEMIHONEST_H__
#include "emp-sh2pc/sh_gen.h"
#include "emp-sh2pc/sh_eva.h"

namespace emp {
block delta_used;
block delta_blocks[8];
template<typename IO>
inline SemiHonestParty<IO>* setup_semi_honest(IO* io, int party, int batch_size = 1024*16) {

	if(party == ALICE) {
		HalfGateGen<IO> * t = new HalfGateGen<IO>(io);
		CircuitExecution::circ_exec = t;
		ProtocolExecution::prot_exec = new SemiHonestGen<IO>(io, t);
		delta_used = t->delta;
	} else {
		HalfGateEva<IO> * t = new HalfGateEva<IO>(io);
		CircuitExecution::circ_exec = t;
		ProtocolExecution::prot_exec = new SemiHonestEva<IO>(io, t);
	}
	return (SemiHonestParty<IO>*)ProtocolExecution::prot_exec;
}

template<typename IO>
inline SemiHonestParty<IO>* setup_semi_honest_mult(IO* io, int party, int tid, int batch_size = 1024*16) {

	if(party == ALICE) {
		HalfGateGen<IO> * t = new HalfGateGen<IO>(io);
		CircuitExecution::circ_exec = t;
		ProtocolExecution::prot_exec = new SemiHonestGen<IO>(io, t);
		delta_blocks[tid] = t->delta;
	} else {
		HalfGateEva<IO> * t = new HalfGateEva<IO>(io);
		CircuitExecution::circ_exec = t;
		ProtocolExecution::prot_exec = new SemiHonestEva<IO>(io, t);
	}
	return (SemiHonestParty<IO>*)ProtocolExecution::prot_exec;
}

inline void finalize_semi_honest() {
	delete CircuitExecution::circ_exec;
	delete ProtocolExecution::prot_exec;
}

}
#endif
