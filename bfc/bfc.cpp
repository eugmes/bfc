#include "AST.h"
#include "Lexer.h"
#include "MLIRGen.h"
#include "Parser.h"
#include "bf/BFPasses.h"

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"

#include "bf/BFDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

using namespace bf;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input bf file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
    overrideTarget("target", cl::desc("Generate code for the given target"),
                   cl::init("default"), cl::value_desc("value"));

namespace {
enum InputType { BF, MLIR };
} // namespace

static cl::opt<InputType> inputType(
    "x", cl::init(BF), cl::desc("The kind of input desired"),
    cl::values(clEnumValN(BF, "bf", "load the input file as a BF source")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as am MLIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMemRefMLIR,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT,
  DumpASM,
};
} // namespace

static cl::opt<Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMemRefMLIR, "mlir-memref",
                          "output the MLIR dump after memref lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")),
    cl::values(clEnumValN(DumpASM, "asm", "output the target assembly")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static std::unique_ptr<bf::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

static void setMLIRDataLayout(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module,
                              llvm::TargetMachine &tm) {
  const auto &dl = tm.createDataLayout();

  auto op = module->getOperation();

  op->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
              mlir::StringAttr::get(&context, dl.getStringRepresentation()));
  op->setAttr(
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
      mlir::StringAttr::get(&context, tm.getTargetTriple().getTriple()));
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(dl, &context);
  op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module,
                    llvm::TargetMachine &tm) {
  // Handle '.bf' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    if (!module)
      return 1;
    setMLIRDataLayout(context, module, tm);
    return 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int loadAndProcessMLIR(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module,
                              llvm::TargetMachine &tm) {
  if (int error = loadMLIR(context, module, tm))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  bool isLoweringToMemRef = emitAction >= Action::DumpMemRefMLIR;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  if (isLoweringToMemRef) {
    pm.addPass(mlir::bf::createBFConvertToMemRef());
    if (enableOpt) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }
  }

  if (isLoweringToLLVM) {
    // Finish lowering the bf IR to the LLVM dialect.
    pm.addPass(mlir::bf::createBFConvertToLLVM());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::LLVM::LLVMFuncOp>();
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    optPM.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    // // FIXME: this is needed to get rid of some unrealized type conversions
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;

  return 0;
}

static int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a BF AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST, llvm::outs());
  return 0;
}

static std::unique_ptr<llvm::Module>
convertToLLVM(llvm::LLVMContext &llvmContext, mlir::ModuleOp module,
              llvm::TargetMachine &tm) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    std::exit(-1);
  }

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0, &tm);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    std::exit(-1);
  }
  return llvmModule;
}

static int runJit(mlir::ModuleOp module) {
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "bf compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::DLTIDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::bf::BFDialect>();

  std::unique_ptr<llvm::TargetMachine> tm;

  if (emitAction == Action::RunJIT) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
      llvm::errs() << "Could not create JITTargetMachineBuilder\n";
      return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
      llvm::errs() << "Could not create TargetMachine\n";
      return -1;
    }

    tm = std::move(tmOrError.get());
  } else {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    std::string targetTriple = overrideTarget;
    if (targetTriple == "default")
      targetTriple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    if (!target) {
      llvm::errs() << error << "\n";
      return -1;
    }

    auto cpu = "generic";
    auto features = "";
    llvm::TargetOptions opts;
    tm.reset(
        target->createTargetMachine(targetTriple, cpu, features, opts, {}));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module, *tm))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->print(llvm::outs());
    return 0;
  }

  if (emitAction == Action::RunJIT) {
    return runJit(*module);
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = convertToLLVM(llvmContext, *module, *tm);
  if (emitAction == Action::DumpLLVMIR) {
    llvm::outs() << *llvmModule << "\n";
    return 0;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CodeGenFileType::AssemblyFile;

  if (tm->addPassesToEmitFile(pass, llvm::outs(), nullptr, fileType)) {
    llvm::errs() << "TargetMachine cannot emit assembly\n";
    return 1;
  }

  pass.run(*llvmModule);
  return 0;
}
