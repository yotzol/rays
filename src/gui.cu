#include "gui.cuh"
#include "window.cuh"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <thread>

// forward declarations
void setup_imgui_style();
void render_ui(Renderer &renderer, Camera &camera);

namespace gui {

void setup_imgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        setup_imgui_style();

        ImGui_ImplGlfw_InitForOpenGL(window::window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
}

void render_imgui(Renderer &renderer, Camera &camera) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        render_ui(renderer, camera);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace gui

void setup_imgui_style() {
        ImGuiStyle &style = ImGui::GetStyle();

        ImVec4 color_dark_bg  = ImVec4(0.05f, 0.05f, 0.1f, 1.0f);
        ImVec4 color_panel_bg = ImVec4(0.1f, 0.1f, 0.15f, 1.0f);

        style.Colors[ImGuiCol_WindowBg]      = color_dark_bg;
        style.Colors[ImGuiCol_ChildBg]       = color_panel_bg;
        style.Colors[ImGuiCol_Text]          = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        style.Colors[ImGuiCol_Border]        = ImVec4(0.2f, 0.2f, 0.25f, 1.0f);
        style.Colors[ImGuiCol_FrameBg]       = ImVec4(0.15f, 0.15f, 0.2f, 1.0f);
        style.Colors[ImGuiCol_Button]        = ImVec4(0.2f, 0.2f, 0.3f, 1.0f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.3f, 0.4f, 1.0f);
        style.Colors[ImGuiCol_ButtonActive]  = ImVec4(0.4f, 0.4f, 0.5f, 1.0f);

        style.WindowPadding  = ImVec2(0, 0);
        style.FramePadding   = ImVec2(4, 3);
        style.ItemSpacing    = ImVec2(8, 4);
        style.WindowRounding = 0.0f;
        style.FrameRounding  = 2.0f;
}

void render_ui(Renderer &renderer, Camera &camera) {
        ImGuiIO &io          = ImGui::GetIO();
        ImVec2 viewport_size = io.DisplaySize;

        // main window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(viewport_size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("RAYS", NULL,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove);
        ImGui::PopStyleVar();

        // two main columns
        ImGui::Columns(2, "MainColumns", true);

        const int bottom_offset = -63;

        // left column: display the rendered scene
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Transparent background
        ImGui::BeginChild("RenderArea", ImVec2(0, bottom_offset), false, ImGuiWindowFlags_NoScrollbar);

        ImVec2 avail_size = ImGui::GetContentRegionAvail();
        ImGui::Image(renderer.gl_texture, avail_size, ImVec2(0, 1), ImVec2(1, 0));

        // camera info display at top-right corner
        ImGui::SetCursorPos(ImVec2(avail_size.x - 190, 10));
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.4f));
        ImGui::BeginChild("CameraInfo", ImVec2(200, 120), true);
        ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "Camera");
        ImGui::Text("X: %f", camera.origin.x);
        ImGui::Text("Y: %f", camera.origin.y);
        ImGui::Text("Z: %f", camera.origin.z);
        ImGui::Text("Yaw:\t%f", camera.yaw);
        ImGui::Text("Pitch:\t%f", camera.pitch);
        ImGui::Text("Roll:\t%f", camera.roll);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::EndGroup();

        ImGui::EndChild();
        ImGui::PopStyleColor();

        // right column: settings panel
        ImGui::NextColumn();
        ImGui::BeginChild("SettingsContent", ImVec2(0, bottom_offset), true);

        // quality section
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.9f, 1.0f, 1.0f));
        ImGui::TextUnformatted("QUALITY SETTINGS");
        ImGui::PopStyleColor();
        ImGui::Separator();

        ImGui::Text("Samples: %d", renderer.config.samples_per_pixel);
        ImGui::SliderInt("##Samples", &renderer.config.samples_per_pixel, 16, 2048);

        ImGui::Text("Bounces: %d", renderer.config.max_depth);
        ImGui::SliderInt("##Bounces", &renderer.config.max_depth, 1, 128);

        static const char *resolutions[] = {"640x360", "1280x720", "1920x1080", "2560x1440", "3840x2160", "1920x1920"};
        ImGui::Text("Resolution");
        ImGui::Combo("##Resolution", &renderer.final_resolution_idx, resolutions, IM_ARRAYSIZE(resolutions));
        ImGui::Spacing();

        // controls section
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.9f, 1.0f, 1.0f));
        ImGui::TextUnformatted("CONTROLS");
        ImGui::PopStyleColor();
        ImGui::Separator();

        ImGui::Text("Movement speed: %f", camera.movement_speed);
        ImGui::SliderFloat("##MovementSpeed", &camera.movement_speed, 0.1, 50);

        ImGui::Spacing();

        // environment section
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.9f, 1.0f, 1.0f));
        ImGui::TextUnformatted("ENVIRONMENT");
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::EndChild();

        // bottom bar
        ImGui::Columns(1);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f, 0.07f, 0.12f, 1.0f));
        ImGui::BeginChild("BottomBar", ImVec2(0, 45), true, ImGuiWindowFlags_NoScrollbar);

        ImGui::SetCursorPos(ImVec2(10, 7));
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.0f, 0.4f, 0.4f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.5f, 0.5f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.0f, 0.6f, 0.6f, 1.0f));

        if (ImGui::Button("RENDER", ImVec2(100, 30))) {
                printf("BUTTON: Export frame clicked.\n");

                if (!window::is_exporting.load()) {
                        printf("\t- Exporting full quality image.\n");
                        window::is_exporting.store(true);

                        std::thread render_thread([&renderer, &camera]() {
                                renderer.render_full_frame("output.png", camera);
                                window::is_exporting.store(false);
                        });
                        render_thread.detach();
                } else {
                        printf("\t- IGNORED: already exporting.\n");
                }
        }

        ImGui::SameLine(120);
        if (ImGui::Button("RENDER VIDEO", ImVec2(100, 30))) {
                printf("BUTTON: Export video clicked.\n");

                if (!window::is_exporting.load()) {
                        printf("\t- Exporting full quality image.\n");
                        window::is_exporting.store(true);

                        std::thread render_thread([&renderer, &camera]() {
                                renderer.render_video("output");
                                window::is_exporting.store(false);
                        });
                        render_thread.detach();
                } else {
                        printf("\t- IGNORED: already exporting.\n");
                }
        }

        ImGui::PopStyleColor(3);

        ImGui::SameLine(230);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.3f, 1.0f));
        if (ImGui::Button("RESET", ImVec2(100, 30))) {
                printf("BUTTON: Reset clicked.\n");
                renderer.render_needs_update = true;
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(340);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.3f, 1.0f));
        if (ImGui::Button("ADD KEYFRAME", ImVec2(100, 30))) {
                printf("BUTTON: Keyframe clicked.\n");
                renderer.camera_positions.push_back(camera);
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(ImGui::GetWindowWidth() - 200);
        ImGui::Text("Rendered Samples: %d", renderer.sample_count);

        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::End();
}
