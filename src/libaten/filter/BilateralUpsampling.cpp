#include <array>
#include "visualizer/atengl.h"
#include "filter/BilateralUpsampling.h"
#include "misc/timer.h"

namespace aten
{
	bool BilateralUpsampling::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		shader::init(width, height, pathVS, pathFS);

		m_hiResNmlDepth.initAsGLTexture(width, height);

		// TODO
		return true;
	}

	void BilateralUpsampling::prepareRender(
		const void* pixels,
		bool revert)
	{
		Blitter::prepareRender(pixels, revert);

		texture::bindAsGLTexture(m_lowResColor, 0, this);		// color low-resolution.
		m_hiResNmlDepth.bindAsGLTexture(1, this);				// nml/depth hi-resolution.
		texture::bindAsGLTexture(m_lowResNmlDepth, 2, this);	// nml/depth low-resolution.
		

		auto hTexel = getHandle("invScreen");
		if (hTexel >= 0) {
			CALL_GL_API(glUniform4f(hTexel, 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f));
		}
	}
}
