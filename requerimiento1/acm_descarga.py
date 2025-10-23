import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os


class ACMDescarga:
    def __init__(self):
        # Usar ruta relativa para la carpeta de descargas
        self.download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "descargas", "acm"))
        os.makedirs(self.download_dir, exist_ok=True)

        # Configuración de Chrome con UC
        options = uc.ChromeOptions()
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # Iniciar Chrome disfrazado
        self.driver = uc.Chrome(options=options)

        # 🔎 Registro de archivos previos para no tocar los de ScienceDirect
        self._bib_previos = set(
            f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")
        )

    def abrir_base_datos(self):
        url = "https://library.uniquindio.edu.co/databases"
        self.driver.get(url)

        wait = WebDriverWait(self.driver, 15)
        wait_long = WebDriverWait(self.driver, 40)

        wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, "onload-background")))

        enlace = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "BASES DATOS x FACULTAD")))
        enlace.click()

        wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, "onload-background")))

        elemento = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@data-content-listing-item='fac-ingenier-a']")))
        elemento.click()

        acm_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@href='https://dl.acm.org/']")))
        acm_link.click()

        self.driver.switch_to.window(self.driver.window_handles[-1])

        # Buscar
        search_box = wait.until(EC.visibility_of_element_located((By.NAME, "AllField")))
        search_box.clear()
        search_box.send_keys('"generative artificial intelligence"')

        search_button = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "button.quick-search__button")))
        search_button.click()

        wait_long.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "li.search__item.issue-item-container")
        ))

        # Cambiar a 50 resultados por página
        link_50 = wait_long.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='per-page separator-end']//a[contains(@href,'pageSize=50')]")
        ))
        self.driver.execute_script("arguments[0].click();", link_50)

        time.sleep(5)  # tiempo para recargar con 50

        # Procesar todas las páginas
        pagina = 0
        while True:
            print(f"📄 Procesando página {pagina + 1}...")

            wait_long.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "li.search__item.issue-item-container")
            ))

            # Seleccionar todos los resultados
            select_all_checkbox = wait_long.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "input[name='markall']")
            ))
            self.driver.execute_script("arguments[0].click();", select_all_checkbox)
            time.sleep(2)

            # Clic en "Export Citations"
            export_button = wait_long.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "a.export-citation")
            ))
            self.driver.execute_script("arguments[0].click();", export_button)
            time.sleep(3)

            # Descargar directamente (BibTeX es el predeterminado)
            download_button = wait_long.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "a.download__btn")
            ))
            self.driver.execute_script("arguments[0].click();", download_button)
            print(f"✅ Página {pagina + 1} descargada.")
            time.sleep(3)

            # 🔒 Cerrar la ventana emergente de exportación
            try:
                close_button = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[title='Close'], button.close, button[data-dismiss='modal']")
                ))
                self.driver.execute_script("arguments[0].click();", close_button)
                print("❌ Ventana de exportación cerrada.")
                time.sleep(2)
            except:
                print("⚠ No se pudo cerrar la ventana, probablemente ya no estaba visible.")

            # Intentar pasar a la siguiente página con el botón "Next"
            try:
                next_button = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "a.pagination__btn--next")
                ))
                self.driver.execute_script("arguments[0].click();", next_button)
                pagina += 1
                time.sleep(5)
            except:
                print("🚀 Fin de resultados.")
                break

        # Al final, unir solo los archivos .bib NUEVOS de esta ejecución
        self.unir_archivos()

    def unir_archivos(self):
        """
        Une SOLO los archivos .bib descargados durante ESTA ejecución
        y borra únicamente esos, dejando intactos los de ScienceDirect.
        """
        # Archivos .bib que hay ahora
        bib_actuales = [f for f in os.listdir(self.download_dir) if f.lower().endswith(".bib")]
        # Tomar solo los nuevos (no presentes antes de iniciar)
        bib_nuevos = [f for f in bib_actuales if f not in self._bib_previos]

        if not bib_nuevos:
            print("ℹ No se detectaron .bib nuevos de ACM en esta ejecución.")
            return

        output_file = os.path.join(self.download_dir, "acmCompleto.bib")

        with open(output_file, "w", encoding="utf-8") as outfile:
            for fname in bib_nuevos:
                with open(os.path.join(self.download_dir, fname), "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")

        print(f"📚 Archivos de ACM unidos en: {output_file}")

        # Borrar SOLO los individuales nuevos de ACM
        for f in bib_nuevos:
            try:
                os.remove(os.path.join(self.download_dir, f))
            except Exception as e:
                print(f"⚠ No se pudo eliminar {f}: {e}")

        print("🧹 Limpiados los .bib individuales nuevos de ACM. Otros archivos permanecen intactos.")

    def cerrar(self):
        self.driver.quit()
