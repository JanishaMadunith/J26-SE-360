# Setup guide

Follow this once per laptop. It takes about an hour, mostly waiting for downloads. Run `make help` at any time to see every command.

## 1. Install the tools

| Tool | Version | Why |
|---|---|---|
| Git | any recent | version control |
| Python | **3.12** | core API, module services, training code |
| Docker Desktop | latest | runs PostgreSQL, Redis and all services together |
| Make | any | short commands (`make up`, `make test`) |
| Flutter SDK | stable channel | mobile app |
| Android Studio | latest | Android SDK, emulator, Kotlin editing |
| JDK | 17 | Android Gradle builds (bundled with Android Studio) |
| VS Code | latest | recommended editor (Android Studio also fine for mobile) |

### macOS

```bash
# Homebrew (skip if installed): https://brew.sh
brew install git python@3.12 make
brew install --cask docker visual-studio-code android-studio
brew install --cask flutter            # or follow https://docs.flutter.dev/get-started/install/macos
```

Open Docker Desktop once so it finishes setup, then check:

```bash
python3.12 --version && docker --version && make --version && flutter --version
```

### Windows 10/11

The simplest path runs the backend inside **WSL2 (Ubuntu)** and Flutter natively on Windows.

1. **WSL2:** in PowerShell as administrator: `wsl --install -d Ubuntu`. Restart, then open "Ubuntu" and create a user.
2. **Docker Desktop:** install from docker.com. In *Settings → Resources → WSL integration*, enable Ubuntu.
3. **Inside Ubuntu (WSL):**
   ```bash
   sudo apt update && sudo apt install -y git make python3.12 python3.12-venv python3-pip
   ```
4. **On Windows (for the mobile app):** install Git for Windows, Android Studio, and Flutter (https://docs.flutter.dev/get-started/install/windows). Add `flutter\bin` to your PATH.
5. **Clone the repo inside WSL** (e.g. `~/projects/J26-SE-360`), not under `/mnt/c/...` — file access is much faster.

If you prefer no WSL: install Python 3.12 from python.org (tick "Add to PATH"), use Git Bash, and install make with `winget install ezwinports.make`.

### Flutter and Android (both OSes)

```bash
flutter doctor                     # follow every suggestion it prints
flutter doctor --android-licenses  # accept all
```

In Android Studio: *SDK Manager → SDK Tools* → tick **Android SDK Command-line Tools**.

Sensors (camera heart rate, typing, app usage, step counter) need a **real Android phone** — emulators can't provide them. On the phone:

1. Settings → About phone → tap *Build number* 7 times to enable Developer options.
2. Developer options → enable **USB debugging**.
3. Connect by USB and run `flutter devices` to check it's listed.

### VS Code extensions

Python, Pylance, **Ruff**, Docker, **Flutter**, **Dart**, Kotlin (fwcd), GitLens, YAML, Markdown Preview Mermaid Support.

## 2. First run of the backend

```bash
git clone https://github.com/OWNER/J26-SE-360.git
cd J26-SE-360
cp .env.example .env      # local settings; never commit .env
make setup                # creates the Python tools, installs pre-commit hooks and the contracts package
make up                   # builds and starts postgres, redis, core-api and m1–m4
make simulate             # sends a fake phone upload and runs the full pipeline
```

Check:

- http://localhost:8000/docs — core API (try `POST /v1/auth/register`, then *Authorize*)
- http://localhost:8000/health — should say `"status": "ok"` with all four modules `true`
- http://localhost:8101/docs … http://localhost:8104/docs — each module service

Stop everything with `make down`. Watch logs with `make logs`.

## 3. First run of the app

```bash
cd mobile/wellness_app
flutter pub get
flutter run                 # with your phone connected
```

The phone must reach the core API on your laptop. Two options:

- **USB (easiest):** `adb reverse tcp:8000 tcp:8000`, then use `http://localhost:8000` as the server URL in the app.
- **Same Wi-Fi:** find your laptop's IP (`ipconfig` / `ifconfig`) and use `http://192.168.x.x:8000`.

See [mobile/wellness_app/README.md](../mobile/wellness_app/README.md) for app-specific steps (permissions, generating platform files).

## 4. Datasets

Datasets are never stored in the repo. Keep them in a folder **next to** the repo:

```
Research/
├── 07-datasets/      ← DATA_DIR (e.g. oulad/, pmdata/, studentlife/, atus/)
└── J26-SE-360/       ← this repo
```

Services read datasets through the `DATA_DIR` environment variable (default `../07-datasets` relative to the repo).

## 5. Common problems

| Problem | Fix |
|---|---|
| `make: command not found` (Windows) | Use WSL, or `winget install ezwinports.make` and reopen Git Bash |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says *running* |
| `port is already allocated` (5432, 8000 …) | Another Postgres or app is using the port. Stop it, or change the port mapping in `.env` / `docker-compose.yml` |
| Core API `/health` shows a module `false` | `make logs` and look at that module's container. Usually a Python error on start-up |
| `alembic` errors after pulling | Run `make migrate` — someone added a database migration |
| Pre-commit fails "schemas out of date" | You edited `contracts/`. Run `make contracts` and commit the regenerated files |
| Scripts fail with `\r` or `bad interpreter` | Windows line endings. `git config --global core.autocrlf input`, then re-clone |
| `flutter doctor`: cmdline-tools missing | Android Studio → SDK Manager → SDK Tools → Android SDK Command-line Tools |
| Gradle: "Unsupported class file major version" | Gradle needs JDK 17. Set Android Studio → Settings → Build Tools → Gradle → Gradle JDK = 17 |
| App can't reach the server | Use `adb reverse tcp:8000 tcp:8000`, or the laptop's LAN IP. Check your firewall allows port 8000 |
| App collects no app-usage data | Grant *Usage access* to the app (Settings → Apps → Special access → Usage access) |
| No typing data | Enable the app's accessibility service (Settings → Accessibility) — see the in-app permissions screen |
