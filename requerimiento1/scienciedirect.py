import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import glob
from dotenv import load_dotenv
from selenium.common.exceptions import TimeoutException, WebDriverException

# Cargar variables del archivo .env
load_dotenv()


class ScienceDirectDescarga:
    def __init__(self, query_text="computational thinking", max_pages=None, per_page=100):
        """
        max_pages: None = todas las páginas; si pasas un número, limitará a ese máximo.
        per_page: cuántos resultados por página en la búsqueda (por defecto 100).
        También puedes usar .env:
            SD_MAX_PAGES=20
            SD_PER_PAGE=100
        """
        # Ruta relativa para las descargas
        self.download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "descargas", "ScienceDirect"))
        os.makedirs(self.download_dir, exist_ok=True)

        # Config desde .env (si no viene por parámetro)
        env_max = os.getenv("SD_MAX_PAGES")
        self.max_pages = int(env_max) if (max_pages is None and env_max and env_max.isdigit()) else max_pages
        self.per_page = int(os.getenv("SD_PER_PAGE", per_page))

        self.query_text = query_text
        self.driver = self._configurar_chrome()

        # 🔎 Snapshot de .bib existentes para no tocarlos (p.ej., acmCompleto.bib)
        self._bib_previos = set(
            f for f in os.listdir(self.download_dir)
            if f.lower().endswith(".bib")
        )
        # Archivos a ignorar en la unión de ScienceDirect
        self._bib_ignorar = {
            "acmcompleto.bib",
            "sciencedirectcompleto.bib",
            "resultado_unificado.bib",
        }

    # ================== UTILIDADES NUEVAS (no rompen tu lógica) ==================

    def _wait_click(self, by, value, timeout=25, desc="elemento"):
        """Espera a que un elemento sea clickable; si falla, reporta y saca screenshot."""
        try:
            w = WebDriverWait(self.driver, timeout, poll_frequency=0.5)
            return w.until(EC.element_to_be_clickable((by, value)))
        except TimeoutException as e:
            print(f"⏳ Timeout esperando {desc}: {by} -> {value} (url: {self.driver.current_url})")
            raise e

    def _wait_presence(self, by, value, timeout=25, desc="elemento"):
        """Espera a que un elemento esté presente en el DOM."""
        try:
            w = WebDriverWait(self.driver, timeout, poll_frequency=0.5)
            return w.until(EC.presence_of_element_located((by, value)))
        except TimeoutException as e:
            print(f"⏳ Timeout esperando presencia de {desc}: {by} -> {value} (url: {self.driver.current_url})")
            raise e

    def _esperar_nuevo_bib(self, timeout=60):
        """Espera a que aparezca al menos un .bib nuevo en download_dir (tras export)."""
        inicio = time.time()
        antes = set(glob.glob(os.path.join(self.download_dir, "*.bib")))
        while time.time() - inicio < timeout:
            time.sleep(1)
            despues = set(glob.glob(os.path.join(self.download_dir, "*.bib")))
            nuevos = despues - antes
            if nuevos:
                return True
        return False

    # ================== CONFIG / LOGIN / NAVEGACIÓN ==================

    def _configurar_chrome(self):
        """Configura Chrome con las opciones optimizadas"""
        options = uc.ChromeOptions()

        # Preferencias de descarga
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0
        }
        options.add_experimental_option("prefs", prefs)

        # Optimizaciones de rendimiento
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        return uc.Chrome(options=options)

    def _codificar_query(self, query):
        """Codificar el query para URL"""
        return query.replace(' ', '%20')

    def acceso_directo_sciencedirect(self):
        """Acceso directo a ScienceDirect usando URL específica (con per_page)."""
        query_codificado = self._codificar_query(self.query_text)
        url_busqueda = (
            f"https://www-sciencedirect-com.crai.referencistas.com/search"
            f"?qs={query_codificado}&show={self.per_page}"
        )

        print(f"🌐 Accediendo directamente a: {url_busqueda}")
        self.driver.get(url_busqueda)

        # Verificar si necesitamos login
        time.sleep(5)
        current_url = self.driver.current_url.lower()

        if "login" in current_url or "signin" in current_url or "accounts.google.com" in current_url:
            print("🔑 Detectada página de login, iniciando autenticación...")
            self.login_google_automatico()
        else:
            print("✅ Acceso directo exitoso - Ya autenticado")

    def login_google_automatico(self):
        """Login automático usando credenciales de entorno (con esperas con nombre)"""
        print("🔑 Iniciando login automático...")
        try:
            google_btn = self._wait_click(By.ID, "btn-google", 25, "btn_google")
            google_btn.click()
            print("✅ Click en botón Google")

            time.sleep(3)

            if "accounts.google.com" in self.driver.current_url:
                print("🔐 Detectada página de Google, ingresando credenciales...")

                email_field = self._wait_click(By.ID, "identifierId", 25, "google_email")
                email_field.send_keys(os.getenv('EMAIL'))
                print("✅ Email ingresado")

                next_btn = self._wait_click(By.ID, "identifierNext", 25, "google_next_email")
                next_btn.click()
                print("✅ Click en siguiente (email)")

                time.sleep(5)
                password_field = self._wait_click(
                    By.XPATH, '//*[@id="password"]/div[1]/div/div[1]/input', 25, "google_password"
                )
                password_field.send_keys(os.getenv('PSWD'))
                print("✅ Contraseña ingresada")

                password_next = self._wait_click(
                    By.XPATH, '//*[@id="passwordNext"]/div/button', 25, "google_next_password"
                )
                password_next.click()
                print("✅ Click en siguiente (contraseña)")

                print("⏳ Esperando a que se complete el login...")
                WebDriverWait(self.driver, 30).until(
                    lambda driver: "sciencedirect.com" in driver.current_url
                )
                print("✅ Login automático completado")
            else:
                print("⚠ No se detectó página de Google, puede que ya esté logueado")
        except Exception as e:
            print(f"❌ Error en login automático: {e}")
            print("💡 Intentando continuar...")

    def obtener_total_paginas(self):
        """Obtener el número total de páginas (con espera robusta)"""
        try:
            time.sleep(8)
            self._wait_presence(By.CSS_SELECTOR, "ol#srp-pagination li", 20, "paginacion")
            elem = self._wait_presence(By.CSS_SELECTOR, "ol#srp-pagination", 20, "paginacion_ol")
            pagination_text = elem.text
            # Normalmente viene algo como "Page 1 of 60"
            parts = pagination_text.split()
            max_pages = parts[-1] if parts else "10"
            if max_pages.isdigit():
                total_paginas = int(max_pages)
                print(f"📊 Total de páginas encontradas: {total_paginas}")
                return total_paginas
            print("⚠ No se pudo obtener el total de páginas, usando valor por defecto (10)")
            return 10
        except Exception as e:
            print(f"❌ Error obteniendo total de páginas: {e}")
            return 10

    # ================== DESCARGA ==================

    def descargar_pagina_actual(self, primera_pagina=False):
        """Descargar los resultados de la página actual con 1 reintento en caso de timeout."""
        for intento in range(2):  # 0 (primer intento) y 1 (reintento)
            try:
                print("📥 Iniciando descarga de página actual...")

                time.sleep(3)

                # Esperar a que el checkbox esté presente
                label = self._wait_click(By.CSS_SELECTOR, "label[for='select-all-results']", 25, "select_all")
                
                # Verificar si el checkbox ya está seleccionado
                checkbox = self.driver.find_element(By.ID, "select-all-results")
                if not checkbox.is_selected():
                    label.click()
                    time.sleep(1)  # Esperar a que se procese el click
                
                print("✅ Checkboxes seleccionados")

                export_btn = self._wait_click(
                    By.XPATH, '//*[@id="srp-toolbar"]/div[1]/span/span[1]/span[2]/div[2]',
                    25, "export_toolbar"
                )
                export_btn.click()
                time.sleep(2)
                print("✅ Modal de exportación abierto")

                export_button = self._wait_click(
                    By.CSS_SELECTOR, "button[data-aa-button='srp-export-multi-bibtex']",
                    25, "export_bibtex"
                )
                export_button.click()
                print("✅ Descarga BibTeX iniciada...")

                # Esperar a que realmente aparezca un nuevo .bib
                if not self._esperar_nuevo_bib(timeout=60):
                    print("⚠ No se detectó nuevo .bib a tiempo (continuo de todos modos)")

                return True

            except TimeoutException:
                print(f"🔁 Reintentando la descarga de la página (intento {intento+2}/2)...")

                try:
                    self.driver.execute_script("location.reload()")
                    time.sleep(5)
                except Exception:
                    pass
                if intento == 1:
                    return False
            except Exception as e:
                print(f"❌ Error descargando página: {e}")
                return False

    def siguiente_pagina(self):
        """Navegar a la siguiente página (con espera robusta)"""
        try:
            next_button = self._wait_click(By.CSS_SELECTOR, 'a[data-aa-name="srp-next-page"]', 20, "next_page")
            self.driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)
            print("✅ Navegando a siguiente página")
            return True
        except Exception as e:
            print(f"❌ No hay más páginas disponibles: {e}")
            return False

    def unir_archivos_bibtex(self):
        """Unir SOLO los .bib NUEVOS de ScienceDirect en uno solo y borrar solo esos."""
        time.sleep(5)  # asegurar fin de descargas

        # Estado actual
        bib_actuales = [
            f for f in os.listdir(self.download_dir)
            if f.lower().endswith(".bib")
        ]

        # .bib nuevos en esta ejecución
        bib_nuevos = [
            f for f in bib_actuales
            if f not in self._bib_previos
               and f.lower() not in self._bib_ignorar
        ]

        if not bib_nuevos:
            print("ℹ No se encontraron .bib nuevos de ScienceDirect para unir.")
            return None

        print(f"📚 Encontrados {len(bib_nuevos)} archivos .bib nuevos de ScienceDirect")

        output_file = os.path.join(self.download_dir, "sciencedirectCompleto.bib")

        with open(output_file, "w", encoding="utf-8") as outfile:
            for fname in bib_nuevos:
                try:
                    with open(os.path.join(self.download_dir, fname), "r", encoding="utf-8") as infile:
                        content = infile.read().strip()
                        if content:
                            outfile.write(content)
                            outfile.write("\n\n")
                    print(f"✅ Procesado: {fname}")
                except Exception as e:
                    print(f"❌ Error procesando {fname}: {e}")

        print(f"📚 Archivos unidos en: {output_file}")

        # Borrar SOLO los .bib nuevos de esta ejecución (no tocamos acmCompleto ni consolidados)
        for f in bib_nuevos:
            try:
                os.remove(os.path.join(self.download_dir, f))
                print(f"🧹 Eliminado: {f}")
            except Exception as e:
                print(f"⚠ No se pudo eliminar {f}: {e}")

        print("✅ Limpieza completada")
        return output_file

    def abrir_base_datos(self):
        """Compatibilidad con tu código existente"""
        print("🔗 Método abrir_base_datos() - Ejecutando descarga...")
        return self.ejecutar_descarga()

    def ejecutar_descarga(self):
        """Método principal para ejecutar toda la descarga (sin tope por defecto)"""
        try:
            print("🚀 Iniciando descarga de ScienceDirect...")

            # Verificar variables de entorno
            email = os.getenv('EMAIL')
            pswd = os.getenv('PSWD')

            if not email or not pswd:
                print("❌ ERROR: Variables de entorno EMAIL y PSWD no configuradas")
                print("💡 Verifica que el archivo .env esté en la carpeta correcta")
                return False

            print(f"✅ Usando email: {email}")

            # Acceso directo
            self.acceso_directo_sciencedirect()

            # Esperar a que cargue completamente
            time.sleep(10)

            # Obtener total de páginas
            total_paginas = self.obtener_total_paginas()
            if total_paginas == 0:
                print("❌ No se encontraron resultados")
                return False

            # Aplicar límite solo si lo configuraste
            if self.max_pages is not None:
                total_paginas = min(self.max_pages, total_paginas)

            print(f"📖 Descargando {total_paginas} páginas...")

            # Descargar cada página
            for pagina_actual in range(total_paginas):
                print(f"📄 Procesando página {pagina_actual + 1} de {total_paginas}...")

                exito = self.descargar_pagina_actual(primera_pagina=(pagina_actual == 0))

                if not exito:
                    print(f"❌ Error en página {pagina_actual + 1}, continuando...")

                # Intentar ir a la siguiente página (excepto en la última)
                if pagina_actual < total_paginas - 1:
                    if not self.siguiente_pagina():
                        break
                time.sleep(2)

            # Unir archivos nuevos de esta ejecución
            archivo_final = self.unir_archivos_bibtex()

            if archivo_final:
                print(f"🎉 Descarga completada exitosamente!")
                print(f"📁 Archivo final: {archivo_final}")
                return True
            else:
                print("❌ Error al unir archivos")
                return False

        except Exception as e:
            print(f"💥 Error crítico durante la descarga: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cerrar(self):
        """Cerrar el navegador de forma segura y silenciosa (evita WinError 6)."""
        try:
            self.driver.quit()
            print("🔚 Navegador cerrado")
        except Exception:
            pass
        finally:
            # Evita que el destructor de uc.Chrome vuelva a intentar quit()
            self.driver = None


# Función de compatibilidad para mantener tu código existente
def download_sciense_articles():
    """Función legacy - usa la nueva clase ScienceDirectDescarga"""
    descargador = ScienceDirectDescarga()
    try:
        return descargador.ejecutar_descarga()
    finally:
        descargador.cerrar()
