; Inno Setup script for Image Analyzer.
; Packages the onedir PyInstaller output (dist\main\*, built via main.spec)
; into a Setup.exe with a Start Menu entry, optional desktop shortcut, and
; a real uninstaller - the Windows equivalent of main.spec's macOS BUNDLE().
;
; Built in CI with: iscc installer\main.iss (run from the repo root, after
; `pyinstaller main.spec` has populated dist\main).

#define MyAppName "Image Analyzer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Acadia AI"
#define MyAppExeName "main.exe"

[Setup]
; Fixed, random GUID identifying this app across versions - do not change,
; or Windows will treat future installer versions as a different app and
; stop offering in-place upgrades.
AppId={{6B2E7B8E-6C36-4C0B-9E7E-6E9B9C7F5C2A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\installer-output
OutputBaseFilename=ImageAnalyzerSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; The whole onedir output (exe + Python runtime + Qt/torch/etc. libs) has to
; ship together - none of it works if split up.
Source: "..\dist\main\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
