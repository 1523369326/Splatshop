#pragma once

#include <memory>
struct SplatEditor;

// 状态、界面和录制集中在独立模块，主编辑器只保留接入点。
class CameraStudio {
public:
	CameraStudio();
	~CameraStudio();
	void open(SplatEditor& editor);
	void close();
	bool isOpen() const;
	bool handleDesktopInput(SplatEditor& editor);
	void render(SplatEditor& editor);
	void drawGui(SplatEditor& editor);

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
