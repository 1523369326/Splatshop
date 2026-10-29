#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "stb/stb_image_write.h"

// 使用已有的 JPEG 编码器写 AVI，无需安装 FFmpeg 或视频插件。
class MjpegAviWriter {
	struct IndexEntry { uint32_t offset, size; };
	std::ofstream file;
	std::vector<IndexEntry> index;
	uint64_t moviSizeOffset = 0, moviStart = 0;
	uint64_t framesOffset = 0, lengthOffset = 0, bufferOffset = 0, streamBufferOffset = 0;
	uint32_t maxFrameBytes = 0;
	int width = 0, height = 0, fps = 30;

	uint64_t position() { return uint64_t(std::streamoff(file.tellp())); }
	void fourcc(const char* code) { file.write(code, 4); }
	void u32(uint32_t value) {
		char bytes[4] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
		file.write(bytes, 4);
	}
	void u16(uint16_t value) {
		char bytes[2] = {char(value), char(value >> 8)};
		file.write(bytes, 2);
	}
	void patch(uint64_t offset, uint32_t value) {
		auto end = file.tellp(); file.seekp(std::streamoff(offset)); u32(value); file.seekp(end);
	}

public:
	~MjpegAviWriter() { close(); }
	bool isOpen() const { return file.is_open(); }
	uint32_t frameCount() const { return uint32_t(index.size()); }

	void open(const std::filesystem::path& path, int w, int h, int rate) {
		if(isOpen()) throw std::runtime_error("A recording is already open.");
		if(w < 16 || h < 16 || w > 4096 || h > 4096 || rate < 1 || rate > 120)
			throw std::runtime_error("Invalid recording resolution or frame rate.");
		width = w; height = h; fps = rate; index.clear(); maxFrameBytes = 0;
		file.open(path, std::ios::binary | std::ios::trunc);
		if(!file) throw std::runtime_error("Cannot open video output file.");
		fourcc("RIFF"); u32(0); fourcc("AVI ");
		fourcc("LIST"); auto hdrlSize = position(); u32(0); fourcc("hdrl");
		fourcc("avih"); u32(56);
		u32(1000000 / fps); u32(0); u32(0); u32(0x10);
		framesOffset = position(); u32(0); u32(0); u32(1);
		bufferOffset = position(); u32(0); u32(width); u32(height);
		for(int i = 0; i < 4; ++i) u32(0);
		fourcc("LIST"); auto strlSize = position(); u32(0); fourcc("strl");
		fourcc("strh"); u32(56); fourcc("vids"); fourcc("MJPG");
		u32(0); u16(0); u16(0); u32(0); u32(1); u32(fps); u32(0);
		lengthOffset = position(); u32(0);
		streamBufferOffset = position(); u32(0); u32(0xffffffff); u32(0);
		u16(0); u16(0); u16(uint16_t(width)); u16(uint16_t(height));
		fourcc("strf"); u32(40); u32(40); u32(width); u32(height);
		u16(1); u16(24); fourcc("MJPG"); u32(width * height * 3);
		u32(0); u32(0); u32(0); u32(0);
		patch(strlSize, uint32_t(position() - strlSize - 4));
		patch(hdrlSize, uint32_t(position() - hdrlSize - 4));
		fourcc("LIST"); moviSizeOffset = position(); u32(0);
		moviStart = position(); fourcc("movi");
		if(!file) { close(); throw std::runtime_error("Cannot write AVI header."); }
	}

	void append(const std::vector<uint8_t>& rgb, int quality) {
		if(!isOpen() || rgb.size() != size_t(width) * height * 3)
			throw std::runtime_error("Invalid video frame.");
		std::vector<uint8_t> jpeg;
		auto callback = [](void* context, void* data, int size) {
			auto& bytes = *static_cast<std::vector<uint8_t>*>(context);
			auto first = static_cast<uint8_t*>(data);
			bytes.insert(bytes.end(), first, first + size);
		};
		if(!stbi_write_jpg_to_func(callback, &jpeg, width, height, 3, rgb.data(), quality))
			throw std::runtime_error("JPEG encoding failed.");
		// 为索引预留空间，在 AVI 1.0 的大小上限前正常结束文件。
		if(position() + jpeg.size() + 9 + 8 + (index.size() + 1) * 16 > 0xf0000000ull)
			throw std::runtime_error("AVI reached its size limit; shorten the path or lower JPEG quality.");
		uint64_t start = position();
		fourcc("00dc"); u32(uint32_t(jpeg.size()));
		file.write(reinterpret_cast<const char*>(jpeg.data()), std::streamsize(jpeg.size()));
		if(jpeg.size() & 1) file.put(0);
		if(!file) throw std::runtime_error("Video write failed; check available disk space.");
		index.push_back({uint32_t(start - moviStart), uint32_t(jpeg.size())});
		maxFrameBytes = std::max(maxFrameBytes, uint32_t(jpeg.size()));
	}

	bool close() {
		if(!isOpen()) return true;
		patch(moviSizeOffset, uint32_t(position() - moviSizeOffset - 4));
		fourcc("idx1"); u32(uint32_t(index.size() * 16));
		for(auto entry : index) {
			fourcc("00dc"); u32(0x10); u32(entry.offset); u32(entry.size);
		}
		patch(4, uint32_t(position() - 8));
		patch(framesOffset, frameCount()); patch(lengthOffset, frameCount());
		patch(bufferOffset, maxFrameBytes); patch(streamBufferOffset, maxFrameBytes);
		file.flush(); bool success = bool(file); file.close(); return success;
	}
};
