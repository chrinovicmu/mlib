#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"   // BinaryOperator, etc.
#include "llvm/IR/IRBuilder.h"      // Helper that inserts new instructions
#include "llvm/IR/Intrinsics.h"     // Intrinsic::ID enum and lookup helpers
#include "llvm/IR/IntrinsicsX86.h"  // x86-specific intrinsic IDs (avx/avx2)
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"   // VectorType, FunctionType, etc.
#include "llvm/IR/Constants.h"      // ConstantFP, ConstantAggregateZero, etc.

// New Pass Manager infrastructure (LLVM 14+)
#include "llvm/Passes/PassBuilder.h"        // PassBuilder, PipelineTuningOptions
#include "llvm/Passes/PassPlugin.h"         // llvmGetPassPluginInfo (entry point)
#include "llvm/IR/PassManager.h"            // FunctionPassManager, AnalysisManager

// Utilities
#include "llvm/Support/raw_ostream.h"       // errs(), outs() — LLVM's stream wrappers
#include "llvm/ADT/SmallVector.h"           // SmallVector<T, N> — stack-allocated vector
#include "llvm/ADT/Statistic.h"             // STATISTIC macro for -stats reporting

using namespace llvm; 

#define DEBUG_TYPE "arith-to-simd"
STATISTIC(FAddLowered, "Number of fadd instructions lowered to AVX2"); 
STATISTIC(FSubLowered,  "Number of fsub instructions lowered to AVX2");                         
STATISTIC(FMulLowered,  "Number of fmul instructions lowered to AVX2");                         
STATISTIC(FDivLowered,  "Number of fdiv instructions lowered to AVX2");

namespace {

/*given llvm scalar tyep f32 or f64, witten the AVX2 256-bit vector type*/ 

static VectorType *getAVX2VectorType(Type *ScalarType, LLVMContext &Ctx){

    if(ScalarType->isFloatTy())
        return VectorType::get(Type::getFloatTy(Ctx), 8, false); 

    if(ScalarType->isDoubleTy())
        return VectorType::get(Type::getFloatTy(Ctx), 4, false); 

    return nullptr; 
}

static Intrinsic::ID getAVX2IntrinsicID(unsigned Opcode, Type *ScalarTy){

    bool isFloat = ScalarTy->isFloatTy(); 
    bool isDouble = ScalarTy->isDoubleTy(); 

    switch(Opcode){

        case Instruction::Fadd:
            if(isFloat) return Intrinsic::X86_avx_add_ps_256; 
            if(isDouble) return Intrinsic:;X86_avx_add_pd_256; 
            break; 
        
        case Instruction::FSub:
            if (isFloat)  return Intrinsic::x86_avx_sub_ps_256; // vsubps ymm
            if (isDouble) return Intrinsic::x86_avx_sub_pd_256; // vsubpd ymm
            break;

        case Instruction::FMul:
            if (isFloat)  return Intrinsic::x86_avx_mul_ps_256; // vmulps ymm
            if (isDouble) return Intrinsic::x86_avx_mul_pd_256; // vmulpd ymm
            break;

        case Instruction::FDiv:
            if (isFloat)  return Intrinsic::x86_avx_div_ps_256; // vdivps ymm
            if (isDouble) return Intrinsic::x86_avx_div_pd_256; // vdivpd ymm
            break;

        default:
            break;

    }

    return Intrinsic::not_intrinsic; 
}

static bool lowerArithToAVX2(Instruction &I){


    //only handle binary operator 
    auto *BO = dyn_cast<BinaryOperator>(&I); 
    if(!BO)
        return false; 

    //only handle flaoting point types 
    Type *ScalarTy = I.getType(); 
    if(!ScalarTy->isFloatTy( && !ScalarTy->isDoubleTy()))
        return false; 

    Intrinsic::ID IID = getAVX2IntrinsicID(I.getOpcode(), ScalarTy); 
    if(IID == Intrinsic::not_intrinsic)
        return false; 

    LLVMContext &Ctx = I.getContext(); 
    VectorType *VecTy = getAVX2VectorType(ScalarTy, Ctx); 
    if(!VecTy)
        return; 

    IRBuilder<> Builder(&I); 
    Builder.setFastMathFlags(BO->getFastMathFlags()); 

    Constant *ZeroVec = ConstantAggregateZero::get(VecTy);

    Value *LaneZero = Builder.getInt32(0); 

    Value *VecA = Builder.CreateInsertElement(
        ZeroVec,
        I.getOperand(0), 
        LaneZero, 
        "simd.a"); 

    Value *VecB  = Builder.CreateInsertElement(
        ZeroVec,
        I.getOperand(1), 
        LaneZero, 
        "simd.a"; 
    ); 

    Module *M = I.getModule(); 
    Function *IntrFn = Intrinsic::getDeclaration(M, IID); 

    SmallVector<Value *, 2> Args = {VecA, VecB}; 
    Value *VecResult = Builder.CreateCall(IntrFn, Args, "simd.res"); 

    Value *ScalarResult = Builder.CreateExtractElement(
        VecResult, 
        LaneZero, 
        "simd.scalar"
    ); 

    I.replaceAllUsesWith(ScalarResult); 
    I.eraseFromParent(); 

    switch(BO->getOpcode()){
        case Instruction::Fadd: ++FAddLowered; break; 
        case Instruction::FSub: ++FSubLowered; break; 
        case Instruction::FMul: ++FMulLowered; break; 
        case Instruction::FDiv: ++FDivLowered; break; 
        default: break 
    }

    return true; 
}

struct ArithToSIMDPass : PassInfoMixin<ArithToSIMDPass> {

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM){

        LLVM_DEBUG(errs() << "[ArithToSIMD] Processing function: "
                          << F.getName() << "\n");

        AttributeSet FnAttrs = F.getAttributes().getFnAttrs(); 
        if(FnAttrs.hasAttribute("target-features")){
            StringRef Features = FnAttrs.getAttributes("target-features")
                .getValueAsString();  

            if(!Features.contains("+avx2")){
                errs() << "[ArithToSIMD] Warning: function '"
                       << F.getName()
                       << "' lacks +avx2 feature string — skipping.\n";
                // Return all-preserved: we didn't touch anything.
                return PreservedAnalyses::all();
            }
        }

        bool Modified = false; 

        SmallVector<instructions *, 32> Worklist; 

        for (BasicBlock &BB : F){
            if(isa<BinaryOperator>(&I)){
                Type *Ty = I.getType; 
                if(Ty->isFloatTy() || Ty->isDoubleTy())
                    Worklist.push_back(&I); 
            }
        }

        for(Instruction *I: Worklist){
            if(lowerArithToAVX2(*I)){
                Modified = true; 
                LLVM_DEBUG(errs() << " lowered: " << *I << "\n"); 

            }
        }

        if(Modified){
            return PreservedAnalyses::none(); 
        }

        return PreservedAnalyses::all(); 
    }

    static StringRef name() {return "arith-to-simd"; }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "ArithToSIMD",
        "v1.0",
        
        [](PassBuilder &PB) {

            PB.registerPipelineParsingCallback(
                [](StringRef Name,
                   FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement> /*Params*/) {
                    if (Name == "arith-to-simd") {
                    
                        FPM.addPass(ArithToSIMDPass());
                        return true;
                    }
                    return false; 
                }
            );
        }
    };
}

