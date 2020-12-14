//----------------------------------------------------------------------------------------------------------------------
// Copyright 2020 Carmelo J Fdez-Aguera
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

#include <cstdint>

// Type safe hangles
template <class Tag, class HandleT = uint32_t>
struct TagHandle
{
	using HandleType = HandleT;

	TagHandle() = default;
	explicit TagHandle(HandleType h)
		: id(h)
	{
	}

	HandleType id{invalidHandle};

	void valid() const { return id != invalidHandle; }
	void invalidate() { id = invalidHandle; }

	static constexpr HandleType invalidHandle = HandleType(-1);
};

struct Eye
{
	nvmath::mat4f worldFromView;
	nvmath::mat4f projection;
};
