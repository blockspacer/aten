#include <array>
#include "visualizer/atengl.h"
#include "filter/BilateralUpsampling.h"
#include "misc/timer.h"

namespace aten
{
	void BilateralUpsampling::prepareRender(
		const void* pixels,
		bool revert)
	{
		Blitter::prepareRender(pixels, revert);

		// Stage 0 texture is binded out of this function.
		texture::bindAsGLTexture(0, 0, this);	// color hi-resolution

		texture::bindAsGLTexture(m_lowResColor, 1, this);
		texture::bindAsGLTexture(m_hiResNmlDepth, 2, this);
		texture::bindAsGLTexture(m_lowResNmlDepth, 3, this);

		auto hTexel = getHandle("invScreen");
		if (hTexel >= 0) {
			CALL_GL_API(glUniform4f(hTexel, 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f));
		}
	}
}
