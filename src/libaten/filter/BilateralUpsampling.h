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

		void setLowResColorTextureHandle(uint32_t handle)
		{
			m_lowResColor = handle;
		}

		void setLowResNmlDepthTextureHandle(uint32_t handle)
		{
			m_lowResNmlDepth = handle;
		}

		uint32_t getHiResNmlDepthTextureHandle()
		{
			return m_hiResNmlDepth.getGLTexHandle();
		}

	private:
		uint32_t m_lowResColor{ 0 };
		uint32_t m_lowResNmlDepth{ 0 };
		texture m_hiResNmlDepth;
	};

}
