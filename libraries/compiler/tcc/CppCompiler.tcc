namespace emll
{
	namespace compiler
	{
		template<typename T>
		void CppCompiler::Emit(Variable& var)
		{
			if (var.IsScalar())
			{
				return EmitScalar<T>(var);
			}
			else if (var.IsVector())
			{
				return EmitVector<T>(var);
			}
			else
			{
				throw new CompilerException(CompilerError::variableTypeNotSupported);
			}
		}

		template<typename T>
		void CppCompiler::EmitScalar(Variable& var)
		{
			switch (var.Scope())
			{
				case VariableScope::Literal:
					EmitLiteral<T>(static_cast<LiteralVar<T>&>(var));
					break;
				case VariableScope::Local:
					if (var.IsVectorRef())
					{
						EmitRef<T>(static_cast<VectorElementVar<T>&>(var));
					}
					else if (var.HasInitValue())
					{
						EmitLocal<T>(static_cast<InitializedScalarVar<T>&>(var));
					}
					else
					{
						EmitLocal<T>(static_cast<ScalarVar<T>&>(var));
					}
					break;

				case VariableScope::Global:
					EmitGlobal<T>(static_cast<InitializedScalarVar<T>&>(var));
					break;

				default:
					throw new CompilerException(CompilerError::variableScopeNotSupported);
			}
		}

		template<typename T>
		void CppCompiler::EmitVector(Variable& var)
		{
			switch (var.Scope())
			{
				case VariableScope::Literal:
					EmitLiteralVector<T>(static_cast<LiteralVarV<T>&>(var));
					break;
				case VariableScope::Global:
					if (var.HasInitValue())
					{
						EmitGlobalVector<T>(static_cast<InitializedVectorVar<T>&>(var));
					}
					else
					{
						EmitGlobalVector<T>(static_cast<VectorVar<T>&>(var));
					}
					break;
				default:
					throw new CompilerException(CompilerError::variableScopeNotSupported);
			}
		}

		template<typename T>
		void CppCompiler::EmitLocal(ScalarVar<T>& var)
		{
			_pfn->Var(var.Type(), var.EmittedName());
		}

		template<typename T>
		void CppCompiler::EmitLocal(InitializedScalarVar<T>& var)
		{
			_pfn->Var<T>(var.EmittedName(), var.Data());
		}

		template<typename T>
		void CppCompiler::EmitLiteral(LiteralVar<T>& var)
		{			
			_pfn->Literal(var.Data());
		}

		template<typename T>
		void CppCompiler::EmitRef(VectorElementVar<T>& var)
		{
			EnsureEmitted(var.Src());
			_pfn->ValueAt(var.Src().EmittedName(), var.Offset());
		}

		template<typename T>
		void CppCompiler::EmitGlobal(InitializedScalarVar<T>& var)
		{
			if (var.IsMutable())
			{
				_module.Global<T>(var.EmittedName(), var.Data());
			}
			else
			{
				_module.Constant<T>(var.EmittedName(), var.Data());
			}
		}

		template<typename T>
		void CppCompiler::EmitLiteralVector(LiteralVarV<T>& var)
		{
			_module.ConstantV<T>(var.EmittedName(), var.Data());
		}

		template<typename T>
		void CppCompiler::EmitGlobalVector(VectorVar<T>& var)
		{
			_module.GlobalV<T>(var.EmittedName(), var.Dimension());
		}

		template<typename T>
		void CppCompiler::EmitGlobalVector(InitializedVectorVar<T>& var)
		{
			_module.GlobalV<T>(var.EmittedName(), var.Data());
		}

		template<typename T>
		void CppCompiler::CompileOutput(const model::OutputNode<T>& node)
		{
			// Output ports have exactly 1 input, output
			auto pInput = node.GetInputPorts()[0];
			Variable* pOutputVar = EnsureVariableFor(node.GetOutputPorts()[0]);
			for (size_t i = 0; i < pInput->Size(); ++i)
			{
				auto outputElt = pInput->GetOutputPortElement(i);
				_pfn->AssignValueAt(pOutputVar->EmittedName(), i, [&outputElt, this](CppFunctionEmitter& fn) {LoadVar(outputElt); });
			}
		}

		template<typename T>
		void CppCompiler::CompileBinary(const nodes::BinaryOperationNode<T>& node)
		{
			auto pInput1 = node.GetInputPorts()[0];
			auto pInput2 = node.GetInputPorts()[1];
			if ((ModelEx::IsPureVector(*pInput1) && ModelEx::IsPureVector(*pInput2)) &&
				!Settings().ShouldUnrollLoops())
			{
				CompileBinaryLoop<T>(node);
			}
			else
			{
				CompileBinaryExpanded<T>(node);
			}
		}

		template<typename T>
		void CppCompiler::CompileBinaryLoop(const nodes::BinaryOperationNode<T>& node)
		{
			Variable* pLVector = EnsureEmitted(node.GetInputPorts()[0]);
			Variable* pRVector = EnsureEmitted(node.GetInputPorts()[1]);
			auto pOutput = node.GetOutputPorts()[0];
			Variable* pResultVector = EnsureEmitted(pOutput);
			auto iVarName = LoopVarName();
			_pfn->BeginFor(iVarName, pOutput->Size());
			{
				_pfn->AssignValueAt(pResultVector->EmittedName(), iVarName);
				_pfn->Op(GetOperator<T>(node),
					[&pLVector, &iVarName, this](CppFunctionEmitter& fn) {_pfn->ValueAt(pLVector->EmittedName(), iVarName); },
					[&pRVector, &iVarName, this](CppFunctionEmitter& fn) {_pfn->ValueAt(pRVector->EmittedName(), iVarName); });
				_pfn->EndStatement();
			}
			_pfn->EndFor();
		}

		template<typename T>
		void CppCompiler::CompileBinaryExpanded(const nodes::BinaryOperationNode<T>& node)
		{
			auto pInput1 = node.GetInputPorts()[0];
			auto pInput2 = node.GetInputPorts()[1];
			auto pOutput = node.GetOutputPorts()[0];
			Variable& resultVar = *(EnsureEmitted(pOutput));
			for (size_t i = 0; i < pInput1->Size(); ++i)
			{
				auto lInput = pInput1->GetOutputPortElement(i);
				auto rInput = pInput2->GetOutputPortElement(i);
				SetVar(resultVar, i);
				_pfn->Op(GetOperator<T>(node), 
					[&lInput, this](CppFunctionEmitter& fn) {LoadVar(lInput); },
					[&rInput, this](CppFunctionEmitter& fn) {LoadVar(rInput); });
				_pfn->EndStatement();
			}
		}

		template<typename T>
		void CppCompiler::CompileSum(const nodes::SumNode<T>& node)
		{
			// SumNode has exactly 1 input and 1 output
			auto input = node.GetInputPorts()[0];
			if (ModelEx::IsPureVector(*input) &&
				!Settings().ShouldUnrollLoops())
			{
				CompileSumLoop<T>(node);
			}
			else
			{
				CompileSumExpanded<T>(node);
			}
		}
		template<typename T>
		void CppCompiler::CompileSumLoop(const nodes::SumNode<T>& node)
		{
			auto pInput = node.GetInputPorts()[0];
			auto pOutput = node.GetOutputPorts()[0];
			Variable* pSrcVector = EnsureEmitted(pInput);
			Variable& resultVar = *(EnsureEmitted(pOutput));
			auto iVarName = LoopVarName();
			_pfn->Assign(resultVar.EmittedName(), GetDefaultForValueType<T>());
			_pfn->BeginFor(LoopVarName(), pSrcVector->Dimension());
			{
				_pfn->Assign(resultVar.EmittedName());
				_pfn->Op(GetAddForValueType<T>(),
					[&resultVar, this](CppFunctionEmitter& fn) {_pfn->Value(resultVar.EmittedName()); },
					[&pSrcVector, &iVarName, this](CppFunctionEmitter& fn) {_pfn->ValueAt(pSrcVector->EmittedName(), iVarName); });
				_pfn->EndStatement();
			}
			_pfn->EndFor();
		}

		template<typename T>
		void CppCompiler::CompileSumExpanded(const nodes::SumNode<T>& node)
		{
			auto pInput = node.GetInputPorts()[0];
			auto pOutput = node.GetOutputPorts()[0];
			Variable& resultVar = *(EnsureEmitted(pOutput));

			_pfn->Assign(resultVar.EmittedName(), GetDefaultForValueType<T>());
			for (size_t i = 0; i < pInput->Size(); ++i)
			{
				auto pRInput = pInput->GetOutputPortElement(i);
				_pfn->Assign(resultVar.EmittedName());
				_pfn->Op(OperatorType::Add,
					[&resultVar, this](CppFunctionEmitter& fn) { _pfn->Value(resultVar.EmittedName()); },
					[&pRInput, this](CppFunctionEmitter& fn) {LoadVar(pRInput); }
				);
				_pfn->EndStatement();
			}
		}

		template<typename T>
		void CppCompiler::CompileBinaryPredicate(const nodes::BinaryPredicateNode<T>& node)
		{
			// Binary predicate has 2 inputs and 1 output
			auto pInput1 = node.GetInputPorts()[0];
			auto pInput2 = node.GetInputPorts()[1];
			auto pOutput = node.GetOutputPorts()[0];
			if (!(ModelEx::IsScalar(*pInput1) && ModelEx::IsScalar(*pInput2)))
			{
				throw new CompilerException(CompilerError::scalarInputsExpected);
			}
			if (!ModelEx::IsScalar(*pOutput))
			{
				throw new CompilerException(CompilerError::scalarOutputsExpected);
			}

			Variable& resultVar = *(EnsureEmitted(pOutput));
			auto lInput = pInput1->GetOutputPortElement(0);
			auto rInput = pInput2->GetOutputPortElement(0);
			_pfn->Assign(resultVar.EmittedName());
			_pfn->Cmp(GetComparison<T>(node),
				[&lInput, this](CppFunctionEmitter& fn) {LoadVar(lInput); },
				[&rInput, this](CppFunctionEmitter& fn) {LoadVar(rInput); });
			_pfn->EndStatement();
		}

		template<typename T>
		void CppCompiler::CompileElementSelectorNode(const model::Node& node)
		{
			auto selectorPort = node.GetInputPorts()[1];
			switch (selectorPort->GetType())
			{
				case model::Port::PortType::Boolean:
					return CompileElementSelector<T, bool>(static_cast<const nodes::ElementSelectorNode<T, bool>&>(node));

				default:
					throw new CompilerException(CompilerError::portTypeNotSupported);
			}
		}

		template<typename T, typename SelectorType>
		void CppCompiler::CompileElementSelector(const nodes::ElementSelectorNode<T, SelectorType>& node)
		{
			auto pElements = node.GetInputPorts()[0];
			if (!ModelEx::IsPureBinary(node))
			{
				// Only support binary right now
				throw new CompilerException(CompilerError::binaryInputsExpected);
			}
			CompileElementSelectorBinary<T, SelectorType>(node);
		}

		///<summary>Compile an element selector node</summary>
		template<typename T, typename SelectorType>
		void CppCompiler::CompileElementSelectorBinary(const nodes::ElementSelectorNode<T, SelectorType>& node)
		{
			auto pElements = node.GetInputPorts()[0];
			auto pSelector = node.GetInputPorts()[1];
			if (!ModelEx::IsScalar(*pSelector))
			{
				throw new CompilerException(CompilerError::scalarInputsExpected);
			}
			auto pOutput = node.GetOutputPorts()[0];
			if (!ModelEx::IsScalar(*pOutput))
			{
				throw new CompilerException(CompilerError::scalarOutputsExpected);
			}
			Variable* pResult = EnsureEmitted(pOutput);
			//
			// If inputs are pure vectors - i.e. all from the same input port, then we can optimize the If, then away into a vector dereference
			//
			//if (ModelEx::IsPureVector(*pElements))
			//{
			//	_fn.Store(pResult, _fn.ValueAt(EnsureEmitted(pElements), pSelectorVal));
			//}
			//else
			{
				auto lVal = pElements->GetOutputPortElement(0);
				auto rVal = pElements->GetOutputPortElement(1);
				_pfn->BeginIf([&pSelector, this](CppFunctionEmitter& fn) { LoadVar(pSelector); });
				{
					_pfn->AssignValue(pResult->EmittedName(), [&lVal, this](CppFunctionEmitter& fn) { LoadVar(lVal); });
				}
				_pfn->EndIf();
				_pfn->BeginElse();
				{
					_pfn->AssignValue(pResult->EmittedName(), [&rVal, this](CppFunctionEmitter& fn) { LoadVar(rVal); });
				}
				_pfn->EndIf();
			}
		}
	}
}