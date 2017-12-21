#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"
#include "texture/texture.h"

namespace aten {
	class BilateralUpsampling : public Blitter {
	public:
		BilateralUpsampling() {}
		virtual ~BilateralUpsampling() {}

	public:
		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;

		void setHiResNmlDepthTextureHandle(uint32_t handle)
		{
			m_hiResNmlDepth = handle;
		}

		uint32_t getLowResNmlDepthTextureHandle()
		{
			return m_lowResNmlDepth.getGLTexHandle();
		}

	private:
		texture m_lowResNmlDepth;
		uint32_t m_hiResNmlDepth{ 0 };
	};

}
