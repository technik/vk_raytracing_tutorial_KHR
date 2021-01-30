//----------------------------------------------------------------------------------------------------------------------
// Copyright 2021 Carmelo J Fdez-Aguera
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <nvmath/nvmath.h>
#include <vector>

class SceneGraph
{
public:
	using mat4f = nvmath::mat4f;

	struct Node
	{
		uint32_t matrixIndex;
		uint16_t childOffset;
		uint16_t numChildren;
	};

	std::vector<Node> nodes;
	std::vector<mat4f> localMatrix;
	std::vector<mat4f> worldMatrix;

	void recalcWorldMatrices()
	{
		size_t i = 0;
		for (auto& node : nodes)
		{
			size_t begin = i + node.childOffset;
			size_t end = begin + node.numChildren;
			++i;
			const auto& parentMtx = worldMatrix[node.matrixIndex];
			for (size_t i = begin; i < end; ++i)
			{
				auto& child = nodes[i];
				const auto& childLocalMtx = localMatrix[child.matrixIndex];
				auto& childWorldMtx = worldMatrix[child.matrixIndex];

				childWorldMtx = parentMtx * childLocalMtx;
			}
		}
	}
};