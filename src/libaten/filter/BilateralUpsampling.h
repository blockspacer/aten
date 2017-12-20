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
		virtual void prepareRender(
			const void* pixels,
			bool revert) override final;

		void setHiResNmlDepthTextureHandle(uint32_t handle)
		{
			m_hiResNmlDepth = handle;
		}

		void setLowResNmlDepthTextureHandle(uint32_t handle)
		{
			m_lowResNmlDepth = handle;
		}

		void setLowResColorTextureHandle(uint32_t handle)
		{
			m_lowResColor = handle;
		}

	private:
		uint32_t m_lowResColor{ 0 };
		uint32_t m_lowResNmlDepth{ 0 };
		uint32_t m_hiResNmlDepth{ 0 };
	};

}
