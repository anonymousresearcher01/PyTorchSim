#ifndef LOWERGEMMINIPASS_H
#define LOWERGEMMINIPASS_H

namespace llvm {
class LowerGemminiPass
    : public PassInfoMixin<LowerGemminiPass> {

public:
  LowerGemminiPass() {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
  static bool isRequired() { return true; }
};
} // namespace llvm

#endif
